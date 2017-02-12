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

## Usage
On a loopback interface with emulated packet loss:

```
$ sudo wping 127.0.0.1
 Seq      RTT     smooth RTT   RTT variation   Packet loss
  1     138 us      138 us         69 us           0%     
  2     152 us      139 us         55 us           0%     
  3     132 us      139 us         43 us           0%     
  4     133 us      138 us         34 us           0%     
  5     115 us      135 us         31 us           0%     
  6     325 us      159 us         70 us           0%     
  7     126 us      155 us         61 us           0%     
  9     251 us      167 us         70 us           11%    
 12     134 us      163 us         61 us           28%    
 13     338 us      184 us         89 us           24%    
 14     246 us      192 us         82 us           21%    
 15     385 us      216 us        110 us           19%    
 16     215 us      216 us         82 us           16%    
 17     362 us      234 us         98 us           14%    
 18     211 us      231 us         79 us           12%    
 19     214 us      229 us         64 us           11%    
 20      92 us      212 us         82 us           10%    
 21     173 us      207 us         71 us           8%
```

The default send interval is 1s and the adaptive packet loss window includes 20 probes (the effects of a packet loss event drop to below 1% after that).

```
Usage:
    target/debug/wping [OPTIONS] [ADDRESS]


positional arguments:
  address               Target ipv4 address

optional arguments:
  -h,--help             show this help message and exit
  -i                    Send interval
  --window WINDOW       Adaptive packet loss calculation for the last N probes
```
