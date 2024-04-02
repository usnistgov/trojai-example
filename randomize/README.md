# Overview

`rewrite.py` is a utility script to rewrite the IP and Ethernet addresses of packets from a packet capture.
The script randomizes the addresses across all sessions/applications. The random address to which the IP and Ethernet addresses are mapped to are consistent within the packets of the same session such that it complies with the network protocol.
The script was primarily created to support TrojAI Cyber Network round to pre-process input network traffic.

# Requirements

To install `pcap-splitter` python library, follow the instructions available here : https://github.com/shramos/pcap-splitter
To install `kamene` python library, follow the instructions available here : https://kamene.readthedocs.io/en/latest/installation.html

# Usage
 
To run the script: `python3 rewrite.py`.
Currently, the paths to the pcaps are hard-coded in the script and can be modified in the `main` method.


