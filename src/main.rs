extern crate argparse;
extern crate pnet;
extern crate pnet_macros_support;
extern crate libc;
extern crate num;
use pnet::packet::ip::{IpNextHeaderProtocols};
use pnet::transport::{ TransportSender, TransportReceiver,
                       transport_channel,ipv4_packet_iter};
use pnet::transport::TransportChannelType::{Layer3};
use pnet::packet::{Packet, MutablePacket};
use pnet::packet::ipv4::{MutableIpv4Packet, Ipv4Flags};
use pnet::packet::icmp::{IcmpPacket, IcmpCode, IcmpTypes};
use pnet::packet::icmp::echo_request::{MutableEchoRequestPacket};
use pnet::packet::icmp::echo_reply::{EchoReplyPacket};
use pnet::packet::{icmp, PacketSize};

// checksum
use pnet_macros_support::types::u16be;

use std::net::{Ipv4Addr, IpAddr};
use std::str::FromStr;
use std::thread;
use std::time::{Duration, Instant};
use std::io::prelude::*;
use std::collections::vec_deque::VecDeque;
use num::pow::pow;
use libc::getpid;

use std::sync::mpsc::{self, RecvTimeoutError};

use argparse::{ArgumentParser, StoreOption};

mod ewma;

use ewma::Ewma;

struct PingResponse {
    nbytes : usize,
    addr : IpAddr,
    seq : u16,
    ttl : u8,
    time : Instant,
}

fn process_responses(mut rx : TransportReceiver,
                sender : mpsc::Sender<PingResponse>) -> ! {
    let mut iter = ipv4_packet_iter(&mut rx);
    loop {
        let res = iter.next();
        let t = Instant::now();
        match res {
            Err (e) => {
                println!("Error receiving packet: {}", e);
            },
            Ok ((ipv4, addr)) => {
                match EchoReplyPacket::new(ipv4.payload()) {
                    Some (echoreply) => {
                        if echoreply.get_icmp_type() == IcmpTypes::EchoReply &&
                            echoreply.get_identifier() == unsafe{getpid() as u16}
                        {
                            // XXX: check addr
                            let resp = PingResponse {
                                nbytes : ipv4.packet_size(),
                                addr : addr,
                                seq : echoreply.get_sequence_number(),
                                ttl : ipv4.get_ttl(),
                                time : t,
                            };
                            if sender.send(resp).is_err() {
                                println!("Internal error: cannot send message to thread")
                            }
                        }
                    },
                    None => ()
                }
            }
        }
    }
}

fn icmp_populate_packet(ipv4 : &mut MutableIpv4Packet, icmp_payload : &Vec<u8>) {
    let mut echo_req = MutableEchoRequestPacket::new(ipv4.payload_mut()).unwrap();
    echo_req.set_icmp_type(IcmpTypes::EchoRequest);
    echo_req.set_icmp_code(IcmpCode::new(0));
    let pid = unsafe {getpid()};
    echo_req.set_identifier(pid as u16);
    echo_req.set_sequence_number(88);
    echo_req.set_payload(&icmp_payload);
}

fn icmp_calc_checksum(ipv4 : &MutableIpv4Packet) -> u16be {
    let icmp = IcmpPacket::new(ipv4.payload()).unwrap();
    icmp::checksum(&icmp)
}

fn icmp_checksum(ipv4 : &mut MutableIpv4Packet) {
    let csum = icmp_calc_checksum(&ipv4);
    let mut ipv4_payload = ipv4.payload_mut();
    let mut echo_req = MutableEchoRequestPacket::new(ipv4_payload).unwrap();
    echo_req.set_checksum(csum);
}

fn icmp_update_seq(ipv4 : &mut MutableIpv4Packet, seq : u16) {
    let mut echo_req = MutableEchoRequestPacket::new(ipv4.payload_mut()).unwrap();
    echo_req.set_sequence_number(seq)
}

fn populate_packet(pkt_buf : &mut [u8], dst : &Ipv4Addr, icmp_payload : Vec<u8>) {
    let mut ipv4 = MutableIpv4Packet::new(pkt_buf).unwrap();

    ipv4.set_next_level_protocol(IpNextHeaderProtocols::Icmp);

    let ipv4_header_len = match MutableIpv4Packet::minimum_packet_size().checked_div(4) {
        Some (l) => l as u8,
        None => panic!("Invalid header len")
    };

    ipv4.set_header_length(ipv4_header_len);
    ipv4.set_version(4);
    ipv4.set_ttl(64);
    ipv4.set_destination(*dst);
    ipv4.set_flags(Ipv4Flags::DontFragment);
    ipv4.set_options(&[]);
    icmp_populate_packet(&mut ipv4, &icmp_payload);
}

fn send_ping (mut tx : TransportSender, dstaddr : String) -> Box<FnMut(u16) -> ()> {
    let dst = match Ipv4Addr::from_str(&dstaddr) {
        Ok (addr) => addr,
        Err (e) => panic!("Couldn't parse address: {}: ", e)
    };

    let icmp_payload = vec![0x41; 60];

    let icmp_len = MutableEchoRequestPacket::minimum_packet_size() +
        icmp_payload.len();
    let total_len = MutableIpv4Packet::minimum_packet_size() + icmp_len;

    let mut pkt_buf : Vec<u8> = vec!(0; total_len);

    populate_packet(&mut pkt_buf, &dst, icmp_payload);
    Box::new(move |seq| {
        let mut ipv4 = MutableIpv4Packet::new(&mut pkt_buf).unwrap();
        icmp_update_seq(&mut ipv4, seq);
        icmp_checksum(&mut ipv4);

        match tx.send_to(ipv4, IpAddr::V4(dst)) {
            Ok (bytes) => if bytes != total_len { panic!("Short send count: {}", bytes) },
            Err (e) => panic!("Could not send: {}", e),
        }
    })
}

#[derive(Debug)]
struct Probe {
    seq : u16,
    sent : Instant,
    received : Option<Instant>,
}

impl Probe {
    fn new(seq : u16, t : Instant) -> Self{
        Probe {
            seq : seq,
            sent : t,
            received : None
        }
    }
    fn received(&mut self, t : Instant) {
        self.received =  Some(t)
    }
    fn rtt(&self) -> Option<Duration> {
        self.received.map(|rx| rx.duration_since(self.sent))
    }
}

#[derive(Debug)]
struct Stats {
    // Window size; can't be too large because we can't (in the general case)
    // have more than 2**16 packets in flight (size of the echoreq seq). Not
    // that is a concern in practice.
    n : u16,

    // Responses to our probes may arrive out of order. But the
    // exponantiated weighted moving average (EWMA) needs to be
    // computed in order. For this reason, we keep around the last
    // Self.n sent probes, so we can go back and recalculate the
    // packet loss EWMA when a probe in the window is responded to
    // (i.e. the packet changes from 'lost' or 'unknown' to 'responded
    // to'. The status of probes that slide out of the window is
    // frozen and added to the packet_loss accumulator.
    ring : VecDeque <Probe>,

    // This field records the packet loss EWMA for the probes that
    // have slid outside our window.
    packet_loss : Option<Ewma>,
}

impl Stats {
    // The 1.0 value for calculating the packet loss
    fn pl_unit() -> u64 {
        1000_000_000
    }
    fn new(n : u16) -> Result<Self, &'static str> {
        if n == 0 {
            Err("Cannot work with an empty window")
        } else {
            Ok(Stats {
                n : n,
                ring : VecDeque::with_capacity(n as usize),
                packet_loss : None,
            })
        }
    }
    fn probe(&mut self, seq : u16, t : Instant) {
        if self.ring.len() == (self.n as usize) {
            let p = self.ring.pop_front().expect("Error popping from non-empty window");
            // Probe slides out of the window, the received/lost
            // designation needs to become final now (if we receive
            // a response to its seq later, there's no way to tell
            // if it's a dup or not, nor can we update the EWMA).
            let lost = match p.received {
                None => Self::pl_unit(),
                Some (_) => 0,
            };
            match self.packet_loss {
                Some (ref mut pl) => {
                    pl.add_sample(lost)
                },
                None => {
                    // This is the first packet that slides out the window
                    assert_eq!(p.seq, 1);
                    self.packet_loss = Some(Ewma::new(lost))
                }
            }
        }
        self.ring.push_back(Probe::new(seq, t))
    }
    fn response(&mut self, seq : u16, t : Instant) {
        match self.ring.iter_mut().find(|p| p.seq == seq) {
            None =>
            // XXX: here we want to be estimating the RTT time for that packet;
            // if the latency went up, the window for the moving average might
            // have become inadequate
                writeln!(&mut std::io::stderr(),
                         "Received seq {} outside the window", seq).unwrap(),
            Some (probe) => probe.received(t)
        }
    }
    fn probe_by_seq(&self, seq : u16) -> Option<&Probe> {
        self.ring.iter().find(|p| p.seq == seq)
    }
    fn estimate_packet_loss(&self, rtt : &Ewma) -> (f64, f64) {
        let now = Instant::now();
        let unit = Self::pl_unit();
        let mut iter = self.ring.iter().filter_map(|p| {
            match p.received {
                Some (_) =>
                    Some(0),
                None => {
                    let elapsed = duration_to_ns(now.duration_since(p.sent));
                    if elapsed > rtt.smoothed + 2 * rtt.variation {
                        // We consider this lost for the current calculation. If
                        // it does arrive later, we'll revise the packet loss percentage
                        // downwards.
                        Some(unit)
                    } else {
                        // Response pending, but reasonably so. Exclude this
                        // probe from the packet loss calculation.
                        None
                    }
                }
            }});
        let mut pl = match self.packet_loss {
            Some (ref pl) => pl.clone(),
            None => {
                // All sent probes are still recorded in the window
                Ewma::new(iter.next().unwrap())
            }
        };
        for s in iter {
            pl.add_sample(s)
        };
        (pl.smoothed as f64 / unit as f64, pl.variation as f64 / unit as f64)
    }
}

fn nanos_to_str(ns : u64) -> String {
    for (i, unit) in ["s", "ms", "us"].into_iter().enumerate() {
        let exp = 9 - i * 3;
        let conv = pow::<u64>(10, exp);
        let res = ns / conv;
        if res != 0 {
            return format!("{} {}", res, unit)
        }
    }
    return format!("{} ns", ns)
}

#[test]
fn test_nanos_to_str() {
    assert_eq!(nanos_to_str(0), "0 ns");
    assert_eq!(nanos_to_str(10), "10 ns");
    assert_eq!(nanos_to_str(1000), "1 us");
    assert_eq!(nanos_to_str(10_000), "10 us");
    assert_eq!(nanos_to_str(10_000_000), "10 ms");
    assert_eq!(nanos_to_str(10_000_000_000), "10 s");
    assert_eq!(nanos_to_str(10_000_000_000_000), "10000 s");
}

fn duration_to_ns(d : Duration) -> u64 {
    d.as_secs() * 1000_000_000 + (d.subsec_nanos() as u64)
}

fn do_probe(probe : &mut Box<FnMut(u16) -> ()>, stats: &mut Stats,
            seq : &mut u16) -> Instant {
    let probe_time = Instant::now();
    *seq += 1;
    probe(*seq);
    stats.probe(*seq, probe_time);
    probe_time
}

fn do_response(rtt_estimate : Option<Ewma> , stats : &mut Stats, resp : PingResponse)
               -> Option<Ewma> {
    stats.response(resp.seq, resp.time);
    match stats.probe_by_seq(resp.seq) {
        None => rtt_estimate, // XXX: outside window
        Some (p) => {
            let rtt_sample = duration_to_ns(p.rtt().unwrap());
            let nrtt = match rtt_estimate {
                None => Ewma::new(rtt_sample),
                Some (mut rtt) => {rtt.add_sample(rtt_sample); rtt},
            };
            let packet_loss = stats.estimate_packet_loss(&nrtt);
            println!("{seq:^5}  {rtt:^9}  {srtt:^10}   {rttvar:^13}   \
                      {spl:^11}",
                     seq=resp.seq, rtt=nanos_to_str(rtt_sample),
                     srtt=nanos_to_str(nrtt.smoothed),
                     rttvar=nanos_to_str(nrtt.variation),
                     spl=format!("{:.0}%", packet_loss.0 * 100.0));
            Some(nrtt)
        }
    }
}

fn main() {
    let mut dest : Option<String> = None;
    let mut opt_interval : Option<String> = None;
    let mut opt_window_size : Option<String> = None;
    {
        let mut parser = ArgumentParser::new();
        parser.refer(&mut dest).add_argument("address", StoreOption, "Target ipv4 address");
        parser.refer(&mut opt_interval).add_option(&["-i"], StoreOption, "Send interval");
        parser.refer(&mut opt_window_size).
            add_option(&["--window"],StoreOption,
                       "Adaptive packet loss calculation for the last N probes");
        match parser.parse_args() {
            Ok (()) => (),
            Err (e) => {
                writeln!(&mut std::io::stderr(), "Error parsing arguments: {}", e).unwrap();
                std::process::exit(2)
            }
        }
    }
    let protocol = Layer3(IpNextHeaderProtocols::Icmp);
    let (tx, rx) = match transport_channel(4096, protocol) {
        Ok ((tx, rx)) => (tx, rx),
        Err (e) => panic!("Could not create the transport channel: {}", e)
    };
    let mut probe = send_ping(tx, dest.expect("Need to supply a destination host"));
    let mut seq = 0;
    let interval = match opt_interval {
        None => Duration::from_millis(1000),
        Some (s) => Duration::from_millis((f64::from_str(&s).unwrap() * 1000.0) as u64)
    };
    let window_size = match opt_window_size {
        None => {
            // It takes a bit over 15 samples after a single packet loss,
            // for the packet loss percentage to drop to below 1%. Default
            // a lange enough value so that there are no abrupt changes in
            // the packet loss percentage when a packet loss event slides
            // outside our window. We still want a window so that the packet
            // loss can go down to zero eventually (though perhaps the .1
            // f64 precision is too much and we can just switch to .0
            // and drop the window logic).
            20 as u16
        },
        Some (sz) => u16::from_str(&sz).expect("Window size must be in the range 1-32768"),
    };
    let mut stats = Stats::new(window_size).expect("Couldn't create ring buffer");
    let mut rtt_estimate : Option<Ewma> = None;

    // Unfortunately, pnet currently only offers a blocking interface, so we
    // have to use a helper thread in order to wait for the next packet OR
    // the expiration of the send timer.
    let (sender, receiver) = mpsc::channel();
    let _ = thread::spawn(move || {
        process_responses(rx, sender)});
    let mut last_probe_time = None;

    // See the format for the data line for the column widths and inter-column
    // whitespace.
    println!(" Seq      RTT     smooth RTT   RTT variation   Packet loss");
    loop {
        let nowish = Instant::now();
        match last_probe_time {
            None => {
                last_probe_time = Some(do_probe(&mut probe, &mut stats, &mut seq));
            },
            Some(t) if nowish.duration_since(t) > interval => {
                last_probe_time = Some(do_probe(&mut probe, &mut stats, &mut seq));
            },
            Some (t) => {
                let timeo = interval - nowish.duration_since(t);
                match receiver.recv_timeout(timeo) {
                    Ok (resp) => {
                        rtt_estimate = do_response(rtt_estimate, &mut stats,
                                                   resp)
                    },
                    Err (RecvTimeoutError::Timeout) => {
                        // Time to probe again
                    },
                    Err (RecvTimeoutError::Disconnected) => {
                        println!("Internal error: receiver thread exited");
                        std::process::exit(1)
                    }
                }
            }
        }
    }
}
