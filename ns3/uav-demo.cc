#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/propagation-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <map>
using namespace ns3;

// ---------------------------------------------------------------------------
// Optional MAC/PHY loss diagnostics (--macStats). Counters per node id.
// ---------------------------------------------------------------------------
struct NodeLossStats
{
    uint64_t txFailed = 0;     // data transmission attempts without ACK
    uint64_t dropRetry = 0;    // MPDU dropped after FrameRetryLimit attempts
    uint64_t dropLifetime = 0; // MPDU expired in the MAC queue (MaxDelay)
    uint64_t dropEnqueue = 0;  // MAC queue full
    uint64_t rxError = 0;      // PSDU received but failed to decode (SINR)
    std::map<std::string, uint64_t> rxDrop; // PHY receive drops by reason
};

static std::map<uint32_t, NodeLossStats> g_lossStats;

static uint32_t
ContextNodeId(const std::string& context)
{
    size_t a = context.find("/NodeList/") + 10;
    return static_cast<uint32_t>(std::stoul(context.substr(a, context.find('/', a) - a)));
}

static void
OnMacDrop(std::string context, WifiMacDropReason reason, Ptr<const WifiMpdu>)
{
    auto& s = g_lossStats[ContextNodeId(context)];
    if (reason == WIFI_MAC_DROP_REACHED_RETRY_LIMIT)
        s.dropRetry++;
    else if (reason == WIFI_MAC_DROP_EXPIRED_LIFETIME)
        s.dropLifetime++;
    else if (reason == WIFI_MAC_DROP_FAILED_ENQUEUE)
        s.dropEnqueue++;
}

static void
OnTxDataFailed(std::string context, Mac48Address)
{
    g_lossStats[ContextNodeId(context)].txFailed++;
}

static void
OnRxError(std::string context, Ptr<const Packet>, double)
{
    g_lossStats[ContextNodeId(context)].rxError++;
}

static void
OnPhyRxDrop(std::string context, Ptr<const Packet>, WifiPhyRxfailureReason reason)
{
    std::ostringstream r;
    r << reason;
    g_lossStats[ContextNodeId(context)].rxDrop[r.str()]++;
}

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

// Read a number for "key": <number>. Returns false if the key is absent or
// not followed by a number.
static bool ParseDoubleField(const std::string &json, const std::string &key, double &out)
{
    size_t k = json.find("\"" + key + "\"");
    if (k == std::string::npos)
        return false;
    size_t colon = json.find(':', k);
    if (colon == std::string::npos)
        return false;
    const char *start = json.c_str() + colon + 1;
    char *end = nullptr;
    double v = strtod(start, &end);
    if (end == start)
        return false;
    out = v;
    return true;
}

// Parse the optional 2-D array "key": [[a, b, ...], [c, d, ...], ...].
// Returns false if the key is absent. If present but malformed, sets `error`.
static bool ParseMatrix(const std::string &json, const std::string &key,
                        std::vector<std::vector<double>> &out, std::string &error)
{
    out.clear();
    size_t k = json.find("\"" + key + "\"");
    if (k == std::string::npos)
        return false;
    size_t colon = json.find(':', k);
    size_t p = (colon == std::string::npos) ? std::string::npos : json.find_first_not_of(" \t", colon + 1);
    if (p == std::string::npos || json[p] != '[')
    {
        error = "\"" + key + "\" is not an array";
        return true;
    }
    p++;  // inside the outer array
    while (true)
    {
        p = json.find_first_not_of(" \t,", p);
        if (p == std::string::npos)
        {
            error = "\"" + key + "\" is not terminated";
            return true;
        }
        if (json[p] == ']')
            return true;  // end of outer array
        if (json[p] != '[')
        {
            error = "\"" + key + "\" must be an array of arrays";
            return true;
        }
        p++;  // inside a row
        std::vector<double> row;
        while (true)
        {
            p = json.find_first_not_of(" \t,", p);
            if (p == std::string::npos)
            {
                error = "\"" + key + "\" row is not terminated";
                return true;
            }
            if (json[p] == ']')
            {
                p++;
                break;
            }
            const char *start = json.c_str() + p;
            char *end = nullptr;
            double v = strtod(start, &end);
            if (end == start)
            {
                error = "\"" + key + "\" contains a non-numeric value";
                return true;
            }
            row.push_back(v);
            p += end - start;
        }
        out.push_back(row);
    }
}

int main(int argc, char *argv[])
{
    // Static ARP is the default. With dynamic ARP every client starts at
    // t = 2 s, the per-node ARP retry timers fire in lock-step (3, 4, 5 s),
    // the un-ACKed broadcast requests collide, and ns-3 marks the entry dead
    // for ArpCache::DeadTimeout (100 s) - longer than the whole run - so the
    // affected users deliver nothing regardless of their radio link.
    // --populateArp=false restores the previous dynamic-ARP behaviour.
    bool populateArp = true;
    CommandLine cmd(__FILE__);
    cmd.AddValue("populateArp", "Pre-populate ARP caches (no ARP exchange)", populateArp);
    bool macStats = false;
    cmd.AddValue("macStats", "Print per-node MAC/PHY loss counters (MACSTATS lines)", macStats);
    double trafficSeconds = 8.0;
    cmd.AddValue("trafficSeconds", "UDP traffic duration in seconds (clients start at 2 s)",
                 trafficSeconds);
    bool printChannel = false;
    cmd.AddValue("printChannel", "Print the Wi-Fi operating channel (CHANNEL line)", printChannel);
    cmd.Parse(argc, argv);
    const double stopTime = 2.0 + trafficSeconds;

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
    // 4b. OPTIONAL OBSTRUCTION LOSS (UAV <-> user)
    // =========================================
    // "obstruction_loss_db": K x N matrix, row k = UAV k, column i = user i
    // (handoff order). Each entry is the loss in dB IN ADDITION to the
    // distance-based LogDistance model (the RF model's loss in excess of
    // free space at the same distance), applied symmetrically.
    // "obstruction_freq_hz": frequency the matrix was computed at; it must
    // match the Wi-Fi operating channel. Without the matrix key the channel
    // below is exactly the original one.
    std::vector<std::vector<double>> obstructionDb;
    std::string obstructionError;
    bool obstruction = ParseMatrix(json, "obstruction_loss_db", obstructionDb, obstructionError);
    double obstructionFreqHz = 0.0;
    if (obstruction)
    {
        if (obstructionError.empty() && (int)obstructionDb.size() != nUav)
        {
            obstructionError = "obstruction_loss_db has " + std::to_string(obstructionDb.size()) +
                               " rows, expected num_uavs=" + std::to_string(nUav);
        }
        for (int k = 0; obstructionError.empty() && k < (int)obstructionDb.size(); k++)
        {
            if ((int)obstructionDb[k].size() != nUser)
            {
                obstructionError = "obstruction_loss_db row " + std::to_string(k) + " has " +
                                   std::to_string(obstructionDb[k].size()) +
                                   " values, expected num_users=" + std::to_string(nUser);
            }
            for (double v : obstructionDb[k])
            {
                if (obstructionError.empty() && !(std::isfinite(v) && v >= 0.0 && v <= 300.0))
                {
                    obstructionError = "obstruction_loss_db values must be finite and in [0, 300] dB";
                }
            }
        }
        if (obstructionError.empty() &&
            !ParseDoubleField(json, "obstruction_freq_hz", obstructionFreqHz))
        {
            obstructionError = "obstruction_loss_db requires \"obstruction_freq_hz\"";
        }
        if (!obstructionError.empty())
        {
            std::cerr << "ERROR: " << obstructionError << std::endl;
            return 2;
        }
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
    if (!obstruction)
    {
        phy.SetChannel(channel.Create());
    }
    else
    {
        // Same chain as YansWifiChannelHelper::Default() (LogDistance with its
        // default attributes + constant-speed delay), followed by a matrix
        // model that adds the per-link obstruction loss. Pairs not in the
        // matrix (user-user, UAV-UAV) get 0 dB extra: the matrix model's own
        // default loss is "infinite" and must be overridden.
        Ptr<YansWifiChannel> ch = CreateObject<YansWifiChannel>();
        Ptr<LogDistancePropagationLossModel> logDistance =
            CreateObject<LogDistancePropagationLossModel>();
        Ptr<MatrixPropagationLossModel> matrix = CreateObject<MatrixPropagationLossModel>();
        matrix->SetDefaultLoss(0.0);
        double sum = 0.0, maxDb = 0.0;
        for (int k = 0; k < nUav; k++)
        {
            for (int i = 0; i < nUser; i++)
            {
                matrix->SetLoss(uav.Get(k)->GetObject<MobilityModel>(),
                                users.Get(i)->GetObject<MobilityModel>(),
                                obstructionDb[k][i],
                                true);
                sum += obstructionDb[k][i];
                maxDb = std::max(maxDb, obstructionDb[k][i]);
            }
        }
        logDistance->SetNext(matrix);
        ch->SetPropagationLossModel(logDistance);
        ch->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());
        phy.SetChannel(ch);
        std::cout << "Obstruction loss: " << nUav << " x " << nUser
                  << " matrix applied (mean " << sum / (nUav * nUser) << " dB, max " << maxDb
                  << " dB, computed at " << obstructionFreqHz / 1e6 << " MHz)" << std::endl;
    }
    phy.Set("TxPowerStart", DoubleValue(30.0));
    phy.Set("TxPowerEnd", DoubleValue(30.0));
    phy.Set("RxSensitivity", DoubleValue(-90.0));

    NetDeviceContainer devices = wifi.Install(phy, wifiMac, allNodes);

    if (obstruction || printChannel)
    {
        Ptr<WifiPhy> wphy = DynamicCast<WifiNetDevice>(devices.Get(0))->GetPhy();
        double channelMhz = wphy->GetFrequency();
        std::cout << "CHANNEL number=" << (int)wphy->GetChannelNumber()
                  << " frequency_mhz=" << channelMhz << " width_mhz=" << wphy->GetChannelWidth()
                  << " band=" << wphy->GetPhyBand() << std::endl;
        if (obstruction && std::abs(obstructionFreqHz / 1e6 - channelMhz) > 1.0)
        {
            std::cerr << "ERROR: obstruction_freq_hz " << obstructionFreqHz / 1e6
                      << " MHz does not match the Wi-Fi operating channel " << channelMhz
                      << " MHz" << std::endl;
            return 2;
        }
    }

    // =========================================
    // 6. INTERNET STACK
    // =========================================
    InternetStackHelper internet;
    internet.Install(allNodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);
    // interface index 0..nUav-1  -> UAVs ;  nUav..  -> users

    if (populateArp)
    {
        NeighborCacheHelper neighborCache;
        neighborCache.PopulateNeighborCache(interfaces);
    }

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
        serverApp.Stop(Seconds(stopTime));
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
        clientApp.Stop(Seconds(stopTime));
    }

    // =========================================
    // 9. FLOW MONITOR
    // =========================================
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    if (macStats)
    {
        const std::string dev = "/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/";
        Config::Connect(dev + "Mac/DroppedMpdu", MakeCallback(&OnMacDrop));
        Config::Connect(dev + "RemoteStationManager/MacTxDataFailed",
                        MakeCallback(&OnTxDataFailed));
        Config::Connect(dev + "Phy/State/RxError", MakeCallback(&OnRxError));
        Config::Connect(dev + "Phy/PhyRxDrop", MakeCallback(&OnPhyRxDrop));
    }

    Simulator::Stop(Seconds(stopTime));
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
            // Report what was sent so a failed flow still counts towards loss.
            std::cout << "Packets Sent: " << flow.second.txPackets << "\n";
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

    if (macStats)
    {
        for (uint32_t n = 0; n < NodeList::GetNNodes(); n++)
        {
            const auto& s = g_lossStats[n];
            std::cout << "MACSTATS node=" << n << " txFailed=" << s.txFailed
                      << " dropRetry=" << s.dropRetry << " dropLifetime=" << s.dropLifetime
                      << " dropEnqueue=" << s.dropEnqueue << " rxError=" << s.rxError;
            for (const auto& [reason, count] : s.rxDrop)
            {
                std::cout << " rxDrop_" << reason << "=" << count;
            }
            std::cout << "\n";
        }
    }

    Simulator::Destroy();
    return 0;
}
