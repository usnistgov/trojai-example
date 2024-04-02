#!/usr/bin/env python3.8

# Copyright (c) 2018-2024.  Peraton Labs Inc.
#
# Permission is hereby granted, free of charge, 
# to any person obtaining a copy of this software 
# and associated documentation files (the "Software"), 
# to deal in the Software without restriction, 
# including without limitation the rights to use, copy, 
# modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons 
# to whom the Software is furnished to do so, subject to 
# the following conditions:

from kamene.all import *
from pathlib import Path
from pcap_splitter.splitter import PcapSplitter
from tqdm import tqdm
import random
import socket
import struct

ETHER_ADDRS_REMAP=dict()
IP_ADDRS_REMAP=dict()
NEW_PKTS=list()
MAC_ADDRS=set()     # Keeps track of all the randomly generated MAC addresses across all sessions
IP_ADDRS=set()      # Keeps track of all the randomly generated IP addresses across all sessions
SEED=42

def pcap2Session(pcapfile,dstdir):

    '''
        Splits a pcap into multiple sessions
    '''

    ps = PcapSplitter(str(pcapfile))
    ps.split_by_session(dstdir)

    return

def genRandomMAC():

    '''
        Generates a random MAC address
        Generated MAC address is unique across all sessions
    '''

    global MAC_ADDRS

    mac_addr="%02x:%02x:%02x:%02x:%02x:%02x" % (random.randint(0,255),
                                                random.randint(0,255),
                                                random.randint(0,255),
                                                random.randint(0,255),
                                                random.randint(0,255),
                                                random.randint(0,255))
    while mac_addr in MAC_ADDRS:
        mac_addr="%02x:%02x:%02x:%02x:%02x:%02x" % (random.randint(0,255),
                                                    random.randint(0,255),
                                                    random.randint(0,255),
                                                    random.randint(0,255),
                                                    random.randint(0,255),
                                                    random.randint(0,255))

    MAC_ADDRS.add(mac_addr)
    return mac_addr

    
def genRandomIP():

    '''
        Generates a random IP address. 
        Generated IP address is unique across all sessions

    '''
    global IP_ADDRS

    ip_addr=socket.inet_ntoa(struct.pack('>I',random.randint(1,0xffffffff)))
    while ip_addr in IP_ADDRS:
        ip_addr=socket.inet_ntoa(struct.pack('>I',random.randint(1,0xffffffff)))
    IP_ADDRS.add(ip_addr)
    return ip_addr

def rewritePkt(pkt):

    '''
        Rewrites the MAC and IP addresses of the packet
        Ensures that the MAC and IP addresses are consistent
        within a given session
    '''

    global NEW_PKTS, ETHER_ADDRS_REMAP, IP_ADDRS_REMAP

    if Ether in pkt:
        mac_s=pkt[0].src
        mac_d=pkt[0].dst
        
        if mac_s not in ETHER_ADDRS_REMAP.keys():
            mac_r=genRandomMAC()
            ETHER_ADDRS_REMAP[mac_s]=mac_r
        pkt[0].src=ETHER_ADDRS_REMAP[mac_s]
        
        if mac_d not in ETHER_ADDRS_REMAP.keys():
            mac_r=genRandomMAC()
            ETHER_ADDRS_REMAP[mac_d]=mac_r
        pkt[0].dst=ETHER_ADDRS_REMAP[mac_d]

    if IP in pkt:
        ip_s=pkt[1].src
        ip_d=pkt[1].dst

        if ip_s not in IP_ADDRS_REMAP.keys():
            ip_r=genRandomIP()
            IP_ADDRS_REMAP[ip_s]=ip_r
        pkt[1].src=IP_ADDRS_REMAP[ip_s]
        
        if ip_d not in IP_ADDRS_REMAP.keys():
            ip_r=genRandomIP()
            IP_ADDRS_REMAP[ip_d]=ip_r
        pkt[1].dst=IP_ADDRS_REMAP[ip_d]

    NEW_PKTS.append(pkt)

def rewritePcap(pcapdir,outdir):

    '''
        Processes packets from each session, 
        rewrites the header addresses and 
        writes the new set of packets to a file
    '''
    
    global NEW_PKTS, ETHER_ADDRS_REMAP, IP_ADDRS_REMAP

    for pcapfile in tqdm(sorted(pcapdir.glob('**/*.pcap'))):
        NEW_PKTS=list()
        ETHER_ADDRS_REMAP=dict()
        IP_ADDRS_REMAP=dict()

        sniff(offline=str(pcapfile),prn=rewritePkt,store=0)
        
        fname=str(pcapfile.stem)
        outfile=outdir.joinpath(f'{fname}.pcap')
        
        wrpcap(str(outfile),NEW_PKTS)

if __name__ == '__main__':

    pcapdir=Path('USTC-TFC2016/')
    pcaps=[
        'Benign/FTP.pcap',
        'Benign/Facetime.pcap',
        'Benign/Gmail.pcap',
        'Benign/Skype.pcap',
        'Malware/Geodo.pcap',
        'Malware/Htbot.pcap',
        'Malware/Shifu.pcap',
        'Malware/Tinba.pcap',
    ]
    sessiondir=Path('sessions')
    sessiondir.mkdir(exist_ok=True)
    rewritedir=Path('rewritten')
    rewritedir.mkdir(exist_ok=True)
    random.seed(SEED)
    for pcap in pcaps:
        print(f'Processing : {pcap}')
        pcapfile=pcapdir.joinpath(pcap)
        
        app=str(pcapfile.stem)
        appdir=sessiondir.joinpath(app)
        appdir.mkdir(exist_ok=True)

        pcap2Session(pcapfile,appdir)
        
        outdir=rewritedir.joinpath(app)
        outdir.mkdir(exist_ok=True)
        
        rewritePcap(appdir,outdir)
