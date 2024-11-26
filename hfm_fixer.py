import FreeSimpleGUI as sg
import pandas as pd
from tabulate import tabulate
import numpy as np
from io import StringIO

sg.theme('SystemDefaultForReal')

col_left = [[sg.Text('Load HFM base map:'),sg.Button('Load',key='-INPUT_MAP-')],
            [sg.Text('Load LOG from ME7 Logger:'),sg.Button('Load',key='-LOG-')],
                     [sg.Button('start',key='-START-')]]
col_right = [[sg.Multiline('',key='-BASE-',size=(95,14),reroute_cprint=True,font={'Courier',8},autoscroll=True)]]

layout = [ [sg.Column(col_left),sg.VerticalSeparator(),sg.Column(col_right)]
            ]

def stick(value, target):
    value = np.array(value)  
    distances = np.abs(value - target) 
    nearest_index = np.argmin(distances)
    nearest_value = value[nearest_index]
    nearest_distance = distances[nearest_index]
    return nearest_value, nearest_distance

def get_rpm_load(input):
    load,rpm = list(map(float,input.columns)),list(map(float,input.index))
    return[rpm,load]

def find_nearest(df,log,rpm,load):
    pass


def main():
    window = sg.Window('Window Title', layout,finalize=True)

    while True:
        event, values = window.read()

        if event == '-START-':
            pass
        elif event == '-INPUT_MAP-':
            base_csv = sg.popup_get_file('Load base map', multiple_files=False)
            base = pd.read_csv(base_csv,index_col=0,dtype=float)
            rpm_load = get_rpm_load(base)
            window['-BASE-'].update('')
            sg.cprint(tabulate(base,headers='keys',tablefmt='psql'),font='Courier 9')
        elif event == '-LOG-':
            log_file = sg.popup_get_file('Load log file',multiple_files=False)
            header_line=-1
            csv_header=''
            csv_lines=list()
            with open(log_file,'rb') as log:
                for i,line in enumerate(log):
                    decoded_line = line.decode('ISO-8859-1',errors='replace')
                    if 'TimeStamp' in decoded_line:
                        csv_header = decoded_line
                        header_line=i
                    if header_line != -1:
                        csv_lines.append(decoded_line)
            csv_lines='\n'.join(csv_lines)
            desired_columns = ["rl_w", "fr_w", "nmot_w"]
            log = pd.read_csv(StringIO(csv_lines)).drop([0,1],axis=0)
            log.columns = log.columns.str.strip()
            log=log[desired_columns].astype(float)
            log['nearest_load'], log['load_distance'] = zip(*log['rl_w'].apply(lambda val: stick(rpm_load[1], val)))
            log['nearest_rpm'], log['rpm_distance'] = zip(*log['nmot_w'].apply(lambda val: stick(rpm_load[0], val)))
            log['rpm_distance']=log['rpm_distance']/1000
            log['load_distance']=log['load_distance']/10
            log['weight'] = 1 / (log['load_distance'] + log['rpm_distance'] + 1) 
            
            interpolated_df = log[['fr_w', 'nmot_w', 'rl_w', 'nearest_rpm', 'nearest_load', 'load_distance','rpm_distance','weight']]

            interpolated_df.to_csv('interp.csv')

        if event == sg.WIN_CLOSED or event == 'end':
            window.close()
            break

        

if __name__ == '__main__':
    main()