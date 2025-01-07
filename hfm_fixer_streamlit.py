import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter
import seaborn as sns


def clean_df(input:pd.DataFrame)    -> pd.DataFrame:
    input.columns.name,input.index.name='rl_w','nmot_w' ##rl_w = load (x-axis), nmot_w = rpm (y-axis)
    input.columns,input.index = input.columns.astype(float).map(lambda x: f"{x:.2f}"),input.index.astype(float).map(lambda x: f"{x:.2f}")
    return(input)

def plot_3d(input:pd.DataFrame):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    X,Y = np.meshgrid(input.columns.astype(float), input.index.astype(float))
    Z = input.values.astype(float)
    plot = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0.1, antialiased=True)
    fig.colorbar(plot, shrink=0.5, aspect=5)
    ax.set_title('KFKHFM Heatmap')
    ax.set_xlabel('rl_w (load)')
    ax.set_ylabel('nmot_w (rpm)')
    ax.set_zlabel('fr_w (%)')
    st.pyplot(fig,use_container_width=False)

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



def clean_log(input:pd.DataFrame,rpm_load:list,desired_columns:list)   -> pd.DataFrame:
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
    input = StringIO(input.getvalue().decode("ISO-8859-1"))
    csv_lines=[]
    for line in input:
        if 'TimeStamp' in line:
            header_line=True
        if header_line is not None:
            csv_lines.append(line)
    csv_lines='\n'.join(csv_lines)     
    return(pd.read_csv(StringIO(csv_lines),dtype=str).drop([0,1],axis=0))

def apply_cmap(val, diff, cmap, norm):
    color = cmap(norm(diff))
    r,g,b,_=color
    r, g, b = [int(255 * c) for c in (r, g, b)]  #scale to 0-255
    brightness = (r * 299 + g * 587 + b * 114) / 1000  #YIQ formula to determine brightness
    text_color = "black" if brightness > 128 else "white" #make text readable on top of background
    return f"background-color: rgba({r},{g},{b},1); color: {text_color}"

def main():
    st.set_page_config(layout='wide',page_title='KFKHFM adjuster tool')
    st.title("KFKHFM adjuster tool")

    st.markdown(
        """
        <style>
        div[data-testid="stDialog"] div[role="dialog"]:has(.big-dialog) {
            width: 890px !important;
            height: 80vh;
        }
        header {visibility: hidden;}
        .streamlit-footer {display: none;}
        div.block-container {padding-top:1rem;}
        </style>
        """, unsafe_allow_html=True)

    desired_columns = ["rl_w", "fr_w", "nmot_w"]
    default_load = ['5','12.5','17.5','25','35','47.5','60','70','80','90','107.5','135','150','160']
    default_rpm = [1000,1500,1750,2000,2250,2500,3000,3500,4000,4500,5000,5500,6000,6500]
    column_left, column_right = st.columns([2,5])
    with column_left:
        uploadedbase = st.file_uploader('Upload KFKHFM base map', type=['csv'],accept_multiple_files=False,key="baseUploader")
        uploadedlog = st.file_uploader('Upload ME7Logger file', type=['csv'],accept_multiple_files=False)
        status = st.container()



    if 'base' not in st.session_state:
        st.session_state.base = pd.DataFrame(1.0001,index=default_rpm,columns=default_load)
        st.session_state.base.index.name = 'nmot_w'
        st.session_state.basecols = pd.DataFrame(st.session_state.base.columns.values.tolist()).transpose()
        st.session_state.basecols.index.name = 'nmot_w'
    if uploadedbase:
        st.session_state.base = pd.read_csv(uploadedbase,dtype=float,index_col=0)
    if 'smooth' not in st.session_state:
        st.session_state.smooth = False
        st.session_state.sigma = 0.0
    if 'log' not in st.session_state:
        st.session_state.log = pd.DataFrame(1.0001,index=default_rpm,columns=desired_columns)
        st.session_state.file = False
        file = False
    if 'output' not in st.session_state:
        st.session_state.output = False



    base = st.session_state.base
    basecols = st.session_state.basecols
    basecols.columns = basecols.values.tolist()
    smooth = st.session_state.smooth
    sigma = st.session_state.sigma
    output = st.session_state.output

    base.columns = st.session_state.basecols.values.flatten()
    with column_right:
        st.subheader("Edit load values:")
        basecols = st.data_editor(basecols,column_config={'nmot_w':st.column_config.NumberColumn(format= '%f',width='small')})
        st.subheader("Edit your KFKHFM Table:")
        base = st.data_editor(base,column_config={'nmot_w':st.column_config.NumberColumn(format= '%f',width='small')},key='out_base')



        if st.button("Apply changes"):
            st.session_state.base = base
            st.session_state.basecols = basecols
            st.rerun()


        if uploadedlog:
            st.session_state.log = uploadedlog.read()
            file=True
            st.session_state.file = file

        file = st.session_state.file
        if file:
            base = clean_df(base)
            log = open_log(uploadedlog)
            log = clean_log(log,get_rpm_load(base),desired_columns)

            interpolated_df = log[['nearest_rpm', 'nearest_load', 'weighted_fr_w', 'weight']].groupby(['nearest_rpm','nearest_load']).sum().reset_index()
            interpolated_df['weighted_fr_w']=interpolated_df['weighted_fr_w']/interpolated_df['weight']

            output = interpolated_df.pivot_table(values='weighted_fr_w',index='nearest_rpm',columns='nearest_load').fillna(1)
            output = clean_df(output)
            output = (base*output).reindex(index=base.index,columns=base.columns).fillna(base)
            st.session_state.output = output
            st.subheader("Output table:")
            cm = sns.color_palette("icefire", as_cmap=True)

            smoothed_data = gaussian_filter(output, sigma=sigma)
            smoothed_df = pd.DataFrame(smoothed_data,index=output.index, columns=output.columns).round(4).astype('float64')
            diff = smoothed_df-base
            max_abs_diff = diff.abs().max().max()
            norm = plt.Normalize(vmin=-max_abs_diff, vmax=max_abs_diff)
            smoothed_df = smoothed_df.map(lambda x: f"{x:.4f}")
            styled_df = smoothed_df.style.format(precision=4).apply(lambda row: [apply_cmap(val, diff_val,cm,norm) for val, diff_val in zip(row, diff.loc[row.name])],axis=1)
            st.markdown('<div id="scroll-target"></div>', unsafe_allow_html=True)
            st.dataframe(styled_df,use_container_width=True)
            st.text('Smooth data if needed')
            sigma = st.slider('Smoothness level',min_value=0.0,max_value=1.0,key='sigma')
            st.text('Paste below to TunerPro')
            st.code(smoothed_df.to_csv(sep='\t',index=False,header=False),language=None)
            status.success('Done! Scroll down to see the output table.', icon='✅')

if __name__ == '__main__':
    main()
