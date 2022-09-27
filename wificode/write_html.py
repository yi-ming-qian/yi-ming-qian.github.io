import os
import glob
import pickle
import numpy as np

data_path = "./experiments/"

f = open(data_path + f'result.html','w')
msg = """<html>
    <head></head>
    <body>"""
figw = 18.9

img = f"./init.png"
tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">init</p>"
msg += tmp

for i in range(20):
    img = f"./iteration-{i}/results/ckpt-200/all_traj.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Iteration {i}</p>"
    msg += tmp

    # tmp = f"<p style=\"font-size: 14pt; text-align: center; margin-right: 30%;\"><b>Example {i}</b></p>"
    # msg += tmp

msg += """</body>
</html>"""
f.write(msg)
f.close()

data_path = "./experiments/"

f = open(data_path + f'vis_dist.html','w')
msg = """<html>
    <head></head>
    <body>"""
figw = 23.9

for i in range(278):
    img = f"./gif/{i}-l1.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Point {i}, L1</p>"
    msg += tmp

    img = f"./gif/{i}-mlp.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Point {i}, MLP</p>"
    msg += tmp

    # tmp = f"<p style=\"font-size: 14pt; text-align: center; margin-right: 30%;\"><b>Example {i}</b></p>"
    # msg += tmp

msg += """</body>
</html>"""
f.write(msg)
f.close()


data_path = "./experiments/"

f = open(data_path + f'localization.html','w')
msg = """<html>
    <head></head>
    <body>"""
figw = 32
folder_names = []
with open('../outputs/' + "folder_list_auto.txt", 'r') as handle:
    for line in handle:
        folder_names.append(line.strip())
dist_all = np.load(data_path+"localization/localize-optimize.npy")
for i, folder_name in enumerate(folder_names):
    print(folder_name)
    img = f"./localization/{folder_name}-singlealign.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">init trajectory</p>"
    msg += tmp

    img = f"./localization/{folder_name}-localization.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">per-point localization {dist_all[i]}</p>"
    msg += tmp

    img = f"./localization/{folder_name}-optimized.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">optimization</p>"
    msg += tmp


msg += """</body>
</html>"""
f.write(msg)
f.close()

data_path = "./experiments/"

f = open(data_path + f'localization1.html','w')
msg = """<html>
    <head></head>
    <body>"""
figw = 30
rssi_error = pickle.load(open('./experiments/evaluate/rssi_error.pickle', "rb"))
wifi_register_error = pickle.load(open('./experiments/evaluate/wifi_register_error.pickle', "rb"))
flp_register_error = pickle.load(open('./experiments/evaluate/flp_register_error.pickle', "rb"))
rssi_error2 = pickle.load(open('./experiments/evaluate/step2/rssi_error.pickle', "rb"))
wifi_register_error2 = pickle.load(open('./experiments/evaluate/step2/wifi_register_error.pickle', "rb"))
folder_names = glob.glob(data_path+"evaluate/*-tango.png")
for i, folder_name in enumerate(folder_names):
    folder_name = os.path.basename(folder_name)
    folder_name = folder_name.replace("-tango.png", "")
    img = f"./evaluate/{folder_name}-tango.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">{folder_name}</p>"
    msg += tmp

    img = f"./evaluate/{folder_name}-wifimap.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">final: wifi localize, RSSI error {rssi_error[folder_name]}</p>"
    msg += tmp

    img = f"./evaluate/{folder_name}-tango-wifi-align.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">final: registration, error {wifi_register_error[folder_name]}</p>"
    msg += tmp

    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"></p>"
    msg += tmp

    img = f"./evaluate/{folder_name}-flp.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">FLP</p>"
    msg += tmp

    img = f"./evaluate/{folder_name}-tango-flp-align.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">FLP: registration, error {flp_register_error[folder_name]}</p>"
    msg += tmp


    

    # img = f"./evaluate/step2/{folder_name}-wifimap.png"
    # tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">step2: wifi localize, RSSI error {rssi_error2[folder_name]}</p>"
    # msg += tmp

    # img = f"./evaluate/step2/{folder_name}-tango-wifi-align.png"
    # tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">step2: registration, error {wifi_register_error2[folder_name]}</p>"
    # msg += tmp


msg += """</body>
</html>"""
f.write(msg)
f.close()


data_path = "./experiments/"

f = open(data_path + f'localization2.html','w')
msg = """<html>
    <head></head>
    <body>"""
figw = 18.9
rssi_error = pickle.load(open('./experiments/evaluate/rssi_error.pickle', "rb"))
wifi_register_error = pickle.load(open('./experiments/evaluate/wifi_register_error.pickle', "rb"))
flp_register_error = pickle.load(open('./experiments/evaluate/flp_register_error.pickle', "rb"))
rssi_error2 = pickle.load(open('./experiments/evaluate/step2/rssi_error.pickle', "rb"))
wifi_register_error2 = pickle.load(open('./experiments/evaluate/step2/wifi_register_error.pickle', "rb"))
folder_names = glob.glob(data_path+"evaluate/*-tango.png")
for i, folder_name in enumerate(folder_names):
    folder_name = os.path.basename(folder_name)
    folder_name = folder_name.replace("-tango.png", "")
    img = f"./evaluate/{folder_name}-tango.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">{folder_name}</p>"
    msg += tmp

    img = f"./evaluate/{folder_name}-wifimap-cslam.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">cslam</p>"
    msg += tmp

    img = f"./evaluate/{folder_name}-flp.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">flp</p>"
    msg += tmp

    img = f"./evaluate/{folder_name}-wifimap-ours_norefine.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">ours (no refine)</p>"
    msg += tmp

    img = f"./evaluate/{folder_name}-wifimap-ours_all.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">ours (all)</p>"
    msg += tmp
    # second line
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"></p>"
    msg += tmp

    img = f"./evaluate/{folder_name}-tango-wifi-align-cslam.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">cslam</p>"
    msg += tmp

    img = f"./evaluate/{folder_name}-tango-flp-align.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">flp</p>"
    msg += tmp

    img = f"./evaluate/{folder_name}-tango-wifi-align-ours_norefine.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">ours (no refine)</p>"
    msg += tmp

    img = f"./evaluate/{folder_name}-tango-wifi-align-ours_all.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">ours (all)</p>"
    msg += tmp


msg += """</body>
</html>"""
f.write(msg)
f.close()