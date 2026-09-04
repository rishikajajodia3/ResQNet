#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
using namespace ns3;

// ---------------------------------------------------------------------------
// Minimal JSON helpers (no external deps) for the ML handoff file
//   data/uav_positions.json :
//     "num_uavs"        : K
//     "uav_positions"   : [[x,y,z], ...]                (K entries)
//     "num_users"       : N
//     "user_positions"  : [[x,y,z], ...]                (N entries)
//     "associations"    : [u0, u1, ...]  (optional)     user i -> UAV index
//
// For K == 1 this program behaves EXACTLY as the original single-UAV
// scenario (same stdout, same topology: one UDP server, all users are
// clients of it). The multi-UAV lines are only printed when K > 1.
// ---------------------------------------------------------------------------

static std::string SlurpNoNewlines(const std::string &path)
{
    std::ifstream f(path);
    if (!f)
    {
        return std::string();
    }
    std::string line, json;
    while (std::getline(f, line))
    {
        json += line;
    }
    return json;
}

// Read an integer value for "key": <int>
static int ParseIntField(const std::string &json, const std::string &key, int fallback)
{
    size_t k = json.find("\"" + key + "\"");
    if (k == std::string::npos)
        return fallback;
    size_t colon = json.find(':', k);
    if (colon == std::string::npos)
        return fallback;
    int v = fallback;
    if (sscanf(json.c_str() + colon + 1, " %d", &v) != 1)
        return fallback;
    return v;
}

// Fill `out` with every [x, y, z] triple inside the array named `key`.
static void ParseTriples(const std::string &json, const std::string &key,
                         std::vector<Vector> &out)
{
    out.clear();
    size_t k = json.find("\"" + key + "\"");
    if (k == std::string::npos)
        return;
    size_t arrStart = json.find('[', k);
    if (arrStart == std::string::npos)
        return;

    // matching close bracket of the outer array (depth counting)
    size_t arrEnd = std::string::npos;
    int depth = 0;
    for (size_t p = arrStart; p < json.size(); p++)
    {
        if (json[p] == '[')
            depth++;
        else if (json[p] == ']')
        {
            depth--;
            if (depth == 0)
            {
                arrEnd = p;
                break;
            }
        }
    }
    if (arrEnd == std::string::npos)
        return;

    size_t p = json.find('[', arrStart + 1);
    while (p != std::string::npos && p < arrEnd)
    {
        double x, y, z;
        if (sscanf(json.c_str() + p + 1, "%lf, %lf, %lf", &x, &y, &z) == 3)
        {
            out.push_back(Vector(x, y, z));
        }
        p = json.find('[', p + 1);
    }
}

// Parse the optional flat integer array "associations": [ ... ].
static void ParseIntArray(const std::string &json, const std::string &key,
                          std::vector<int> &out)
{
    out.clear();
    size_t k = json.find("\"" + key + "\"");
    if (k == std::string::npos)
        return;
    size_t arrStart = json.find('[', k);
    size_t arrEnd = json.find(']', arrStart);
    if (arrStart == std::string::npos || arrEnd == std::string::npos)
        return;
    size_t p = arrStart + 1;
    while (p < arrEnd)
    {
        // skip whitespace / commas
        while (p < arrEnd && (json[p] == ' ' || json[p] == ',' || json[p] == '\t'))
            p++;
        if (p >= arrEnd)
            break;
        int v = 0;
        int consumed = 0;
        if (sscanf(json.c_str() + p, "%d%n", &v, &consumed) == 1)
        {
            out.push_back(v);
            p += consumed;
        }
        else
        {
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    // =========================================
    // 1. READ ML HANDOFF
    // =========================================
    std::string json = SlurpNoNewlines("data/uav_positions.json");
    if (json.empty())
    {
        std::cerr << "Could not open data/uav_positions.json" << std::endl;
        return 1;
    }

    std::vector<Vector> uavPos;
    std::vector<Vector> userPos;
    std::vector<int> assoc;
    ParseTriples(json, "uav_positions", uavPos);
    ParseTriples(json, "user_positions", userPos);
    ParseIntArray(json, "associations", assoc);

    int nUav = ParseIntField(json, "num_uavs", (int)uavPos.size());
    int nUser = ParseIntField(json, "num_users", (int)userPos.size());

    if (uavPos.empty() || (int)uavPos.size() != nUav)
    {
        std::cerr << "Could not parse " << nUav << " UAV positions from data/uav_positions.json"
                  << std::endl;
        return 1;
    }

    // legacy single-UAV compatibility line (kept for every K; the
    // orchestrator's existing check looks for UAV 0)
    std::cout << "ML UAV Position: ("
              << uavPos[0].x << ", " << uavPos[0].y << ", " << uavPos[0].z << ")"
              << std::endl;

    bool multi = nUav > 1;
    if (multi)
    {
        std::cout << "ML UAV Positions: " << nUav << std::endl;
        for (int k = 0; k < nUav; k++)
        {
            std::cout << "UAV " << (k + 1) << ": ("
                      << uavPos[k].x << ", " << uavPos[k].y << ", " << uavPos[k].z << ")"
                      << std::endl;
        }
    }

    // =========================================
    // 2. NODES
    // =========================================
    NodeContainer uav;
    uav.Create(nUav);
    NodeContainer users;
    users.Create(nUser);

    // =========================================
    // 3. MOBILITY / POSITIONS
    // =========================================
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    Ptr<ListPositionAllocator> uavAlloc = CreateObject<ListPositionAllocator>();
    for (int k = 0; k < nUav; k++)
    {
        uavAlloc->Add(uavPos[k]);
    }
    mobility.SetPositionAllocator(uavAlloc);
    mobility.Install(uav);

    // Ground users: the SAME positions Sionna RT / PPO evaluated.
    Ptr<ListPositionAllocator> userAlloc = CreateObject<ListPositionAllocator>();
    uint32_t usersLoaded = 0;
    for (int i = 0; i < nUser && i < (int)userPos.size(); i++)
    {
        userAlloc->Add(userPos[i]);
        usersLoaded++;
    }
    if ((int)usersLoaded != nUser)
    {
        std::cerr << "WARNING: expected " << nUser
                  << " entries in \"user_positions\", found " << usersLoaded
                  << "; using generated fallback positions." << std::endl;
        userAlloc = CreateObject<ListPositionAllocator>();
        for (int i = 0; i < nUser; i++)
        {
            userAlloc->Add(Vector(uavPos[0].x + i * 5.0, uavPos[0].y + i * 3.0, 0.0));
        }
    }
    else
    {
        std::cout << "Loaded " << usersLoaded
                  << " ground-user positions from data/uav_positions.json" << std::endl;
    }
    mobility.SetPositionAllocator(userAlloc);
    mobility.Install(users);

    // =========================================
    // 4. ASSOCIATION (user -> UAV)
    // =========================================
    // Use the association from the handoff file (best-RSS server chosen by
    // Sionna). If it is absent or malformed, fall back to nearest UAV by
    // 3-D Euclidean distance.
    bool assocFromFile = ((int)assoc.size() == nUser);
    if (!assocFromFile)
    {
        assoc.assign(nUser, 0);
        for (int i = 0; i < nUser; i++)
        {
            double best = 1e300;
            int bestK = 0;
            Vector up = (i < (int)userPos.size())
                            ? userPos[i]
                            : Vector(uavPos[0].x + i * 5.0, uavPos[0].y + i * 3.0, 0.0);
            for (int k = 0; k < nUav; k++)
            {
                double dx = up.x - uavPos[k].x;
                double dy = up.y - uavPos[k].y;
                double dz = up.z - uavPos[k].z;
                double d = dx * dx + dy * dy + dz * dz;
                if (d < best)
                {
                    best = d;
                    bestK = k;
                }
            }
            assoc[i] = bestK;
        }
    }
    for (int i = 0; i < nUser; i++)
    {
        if (assoc[i] < 0 || assoc[i] >= nUav)
            assoc[i] = 0;
    }
    if (multi)
    {
        std::cout << "Associations: " << (assocFromFile ? "from handoff (best-RSS server)"
                                                        : "computed (nearest UAV)")
                  << std::endl;
    }

    // =========================================
    // 5. WIFI  (single shared 802.11n ad-hoc channel)
    // =========================================
    NodeContainer allNodes;
    allNodes.Add(uav);
    allNodes.Add(users);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);

    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    phy.Set("TxPowerStart", DoubleValue(30.0));
    phy.Set("TxPowerEnd", DoubleValue(30.0));
    phy.Set("RxSensitivity", DoubleValue(-90.0));

    NetDeviceContainer devices = wifi.Install(phy, wifiMac, allNodes);

    // =========================================
    // 6. INTERNET STACK
    // =========================================
    InternetStackHelper internet;
    internet.Install(allNodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);
    // interface index 0..nUav-1  -> UAVs ;  nUav..  -> users

    if (multi)
    {
        for (int k = 0; k < nUav; k++)
        {
            std::cout << "UAV " << (k + 1) << " address: "
                      << interfaces.GetAddress(k) << std::endl;
        }
    }

    // =========================================
    // 7. ONE UDP SERVER PER UAV
    // =========================================
    uint16_t port = 9000;
    for (int k = 0; k < nUav; k++)
    {
        UdpServerHelper server(port);
        ApplicationContainer serverApp = server.Install(uav.Get(k));
        serverApp.Start(Seconds(1.0));
        serverApp.Stop(Seconds(10.0));
    }

    // =========================================
    // 8. UDP CLIENT PER USER -> ITS ASSOCIATED UAV
    // =========================================
    for (int i = 0; i < nUser; i++)
    {
        Ipv4Address target = interfaces.GetAddress(assoc[i]); // UAV interface
        UdpClientHelper client(target, port);
        client.SetAttribute("MaxPackets", UintegerValue(1000));
        client.SetAttribute("Interval", TimeValue(MilliSeconds(10)));
        client.SetAttribute("PacketSize", UintegerValue(512));
        ApplicationContainer clientApp = client.Install(users.Get(i));
        clientApp.Start(Seconds(2.0));
        clientApp.Stop(Seconds(10.0));
    }

    // =========================================
    // 9. FLOW MONITOR
    // =========================================
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    Simulator::Stop(Seconds(10.0));
    Simulator::Run();

    // =========================================
    // 10. RESULTS  (format unchanged)
    // =========================================
    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    auto stats = monitor->GetFlowStats();

    std::cout << "\n====================================\n";
    std::cout << "       UAV NETWORK RESULTS\n";
    std::cout << "====================================\n";

    for (const auto &flow : stats)
    {
        Ipv4FlowClassifier::FiveTuple tuple = classifier->FindFlow(flow.first);

        std::cout << "\nFlow: " << tuple.sourceAddress << " -> "
                  << tuple.destinationAddress << "\n";

        if (flow.second.rxPackets == 0)
        {
            std::cout << "No packets received.\n";
            continue;
        }

        double duration = flow.second.timeLastRxPacket.GetSeconds() -
                          flow.second.timeFirstTxPacket.GetSeconds();
        double throughput = 0.0;
        if (duration > 0)
        {
            throughput = flow.second.rxBytes * 8.0 / duration / 1000000.0;
        }
        double delay = flow.second.delaySum.GetSeconds() / flow.second.rxPackets * 1000.0;
        double packetLoss = 100.0 * (flow.second.txPackets - flow.second.rxPackets) /
                            flow.second.txPackets;

        std::cout << "Packets Sent: " << flow.second.txPackets << "\n";
        std::cout << "Packets Received: " << flow.second.rxPackets << "\n";
        std::cout << "Throughput: " << throughput << " Mbps\n";
        std::cout << "Average Delay: " << delay << " ms\n";
        std::cout << "Packet Loss: " << packetLoss << " %\n";
    }

    std::cout << "\n====================================\n";

    Simulator::Destroy();
    return 0;
}
