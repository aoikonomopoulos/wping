# wping

`wping` is similar to `ping`, except it tries to produce more useful output

## Motivation
Estimating packet loss with `ping` usually involves either eyeballing
the output or developing an intuition for the percentage of lost
packets by looking at the timing of the output. Alternatively, one can
run it in a loop and observe the cumulative statistics that `ping`
prints out when it exits, then try to follow them over time. We can do
better.

## How wping works

`wping` calculates and displays an exponential weighted moving average
(EWMA) of the round-trip time and apparent packet loss. For the
purposes of calculating the packet loss, a packet is considered lost
if no response has been received within smoothed(RTT) + 2 *
variation(RTT). If, however, a response arrives for a previously sent
packet, `wping` will revise the packet loss upward.

When the apparent packet loss drops below 1% (i.e. when packet losses
are far enough in the past), the packet loss is reported as 0%.

The EWMA calculation is performed according to RFC6298.

Updated statistics are printed once per send interval: when a response
is received or, failing that, when the next probe is about to be sent.

## Installation

Currently, the way to install `wping` is via cargo. If cargo is not
avaliable as a distribution package, the easiest way to get it is by
installing [rustup](https://www.rustup.rs) and then:

```
$ cargo install wping
$ sudo setcap cap_net_raw+ep ~/.cargo/bin/wping
```

## Usage
On an interface with emulated packet loss:

```
$ wping en.wikipedia.org
PING 91.198.174.192 for en.wikipedia.org (91.198.174.192)
 Seq      RTT      smooth RTT   RTT variation   Packet loss
  1      96 ms       96 ms          48 ms           0%
  2      95 ms       96 ms          36 ms           0%
  3     108 ms       97 ms          30 ms           0%
  4     100 ms       98 ms          23 ms           0%
  -        -         98 ms          23 ms           12%
  6      97 ms       97 ms          17 ms           11%
  7      95 ms       97 ms          14 ms           10%
  8      95 ms       97 ms          10 ms           8%
  9      96 ms       97 ms          8 ms            7%
 10      97 ms       97 ms          6 ms            6%
 11      98 ms       97 ms          5 ms            6%
 12      97 ms       97 ms          3 ms            5%
  -        -         97 ms          3 ms            17%
 14      96 ms       97 ms          3 ms            15%
 15     108 ms       98 ms          5 ms            13%
 16      97 ms       98 ms          4 ms            11%
 17      97 ms       98 ms          3 ms            10%
  -        -         98 ms          3 ms            21%
 19      96 ms       98 ms          3 ms            18%
 20     111 ms       99 ms          5 ms            16%
 21      96 ms       99 ms          5 ms            14%
```

The default send interval is 1s and the adaptive packet loss window
includes 20 probes (the effects of a packet loss event drop to below
1% after that). At startup, `wping` prints out all IP addresses
retrieved for the target domain as a gentle reminder that round-robin
DNS might be at play.

```
Usage:
    wping [OPTIONS] [ADDRESS]


positional arguments:
  address               Target hostname or IPv4 address

optional arguments:
  -h,--help             show this help message and exit
  -i                    Send interval
  -s,--packet-size PACKET_SIZE
                        Payload size in bytes
  --window WINDOW       Adaptive packet loss calculation for the last N probes
  -x,--extended         Include additional information in the output
```
