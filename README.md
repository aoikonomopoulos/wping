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

## Usage
On a loopback interface with emulated packet loss:

```
$ sudo wping 127.0.0.1
 Seq      RTT      smooth RTT   RTT variation   Packet loss
  1     315 us       315 us        157 us           0%
  -        -         315 us        157 us           12%
  3     335 us       318 us        123 us           11%
  4     346 us       321 us         99 us           10%
  5     339 us       323 us         79 us           8%
  6     131 us       299 us        107 us           7%
  7     128 us       278 us        123 us           6%
  8     322 us       283 us        103 us           6%
  -        -         283 us        103 us           17%
 10     134 us       265 us        115 us           15%
 11     332 us       273 us        103 us           13%
  -        -         273 us        103 us           24%
 13     138 us       256 us        111 us           21%
 14     315 us       264 us         98 us           18%
  -        -         264 us         98 us           29%
 16     139 us       248 us        104 us           25%
 17     324 us       258 us         97 us           22%
 18     126 us       241 us        106 us           19%
 19     124 us       226 us        108 us           17%
 20     309 us       237 us        102 us           15%
 21     318 us       247 us         96 us           13%
```

The default send interval is 1s and the adaptive packet loss window
includes 20 probes (the effects of a packet loss event drop to below
1% after that). On Linux, you can avoid the need for sudo by setting
the appropriate capability on the executable:

```
# setcap cap_net_raw+ep /path/to/wping
```

```
Usage:
    wping [OPTIONS] [ADDRESS]


positional arguments:
  address               Target ipv4 address

optional arguments:
  -h,--help             show this help message and exit
  -i                    Send interval
  --window WINDOW       Adaptive packet loss calculation for the last N probes
  -x,--extended         Include additional information in the output
```
