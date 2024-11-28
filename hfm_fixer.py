import FreeSimpleGUI as sg
import pandas as pd
from tabulate import tabulate
import numpy as np
from io import StringIO

sg.theme('SystemDefaultForReal')
desired_columns = ["rl_w", "fr_w", "nmot_w"]

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

def get_rpm_load(input:pd.DataFrame) -> list:
    load,rpm = list(map(float,input.columns)),list(map(float,input.index))
    return[rpm,load]

def clean_df(input:pd.DataFrame)    -> pd.DataFrame:
    input.columns.name,input.index.name='rl_w','nmot_w' ##rl_w = load (x-axis), nmot_w = rpm (y-axis)
    input.columns,input.index = input.columns.astype(float).map(lambda x: f"{x:.2f}"),input.index.astype(float).map(lambda x: f"{x:.2f}")
    return(input)

def clean_log(input:pd.DataFrame,rpm_load:list)   -> pd.DataFrame:
    input.columns = input.columns.str.strip()
    input=input[desired_columns].astype(float)
    input['nearest_load'], input['load_distance'] = zip(*input['rl_w'].apply(lambda val: stick(rpm_load[1], val)))
    input['nearest_rpm'], input['rpm_distance'] = zip(*input['nmot_w'].apply(lambda val: stick(rpm_load[0], val)))
    input['rpm_distance']=input['rpm_distance']/1000
    input['load_distance']=input['load_distance']/10
    input['weight'] = 1 / (input['load_distance'] + input['rpm_distance'] + 1) 
    input['weighted_fr_w']=input['fr_w']*input['weight']
    return(input)

def open_log(input):
    header_line=None
    csv_lines=[]
    with open(input,'rb') as log:
        for i,line in enumerate(log):
            decoded_line = line.decode('ISO-8859-1',errors='replace')
            if 'TimeStamp' in decoded_line:
                header_line=i
            if header_line is not None:
                csv_lines.append(decoded_line)
        csv_lines='\n'.join(csv_lines)     
    return(pd.read_csv(StringIO(csv_lines)).drop([0,1],axis=0))


def main():
    window = sg.Window('Window Title', layout,finalize=True)

    while True:
        event, values = window.read()

        if event == '-START-':
            pass
        elif event == '-INPUT_MAP-':
            base_csv = sg.popup_get_file('Load base map', multiple_files=False)
            base = pd.read_csv(base_csv,index_col=0,dtype=float)
            base = clean_df(base)
            
            window['-BASE-'].update('')
            sg.cprint(tabulate(base,headers='keys',tablefmt='psql'),font='Courier 9')
        elif event == '-LOG-':
            log_file = sg.popup_get_file('Load log file',multiple_files=False)
            log = open_log(log_file)
            log = clean_log(log,get_rpm_load(base))

            interpolated_df = log[['nearest_rpm', 'nearest_load', 'weighted_fr_w', 'weight']].groupby(['nearest_rpm','nearest_load']).sum().reset_index()
            interpolated_df['weighted_fr_w']=interpolated_df['weighted_fr_w']/interpolated_df['weight']
            output = interpolated_df.pivot_table(values='weighted_fr_w',index='nearest_rpm',columns='nearest_load').fillna(1)
            output = clean_df(output)
            print('logs:')
            print(output.reindex(base.index).round(2))
            print('outcome:')
            print((base*output.reindex(base.index).fillna(base)).round(2))

        if event == sg.WIN_CLOSED or event == 'end':
            window.close()
            break

        

if __name__ == '__main__':
    main()