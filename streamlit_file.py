#------------------------------Importing Libraries----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------#
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io



###-----set page title----------------------------------------------------------------------------##
st.set_page_config(page_title="ICG Signal Processor", layout="wide")

# Title and description
st.title("ICG Signal Processing App")
st.markdown("""
This app processes Impedance Cardiography (ICG) data and visualizes the filtered signals.
Upload your CSV file and adjust the parameters to analyze the ICG waveform.
""")

##-----set sidebar parameters---------------------------------------------------------------------##
st.sidebar.image("logo.png", width = 120)
st.sidebar.header("Upload and Parameters setting")


# File upload
uploaded_file = st.sidebar.file_uploader("Choose a TXT or CSV file", type=["txt", "csv"])

# Parameters section
st.sidebar.subheader("Processing Parameters")

# Sampling frequency
fs = st.sidebar.number_input("Sampling Frequency (Hz)", min_value=1, value=62, step=1)

# Filter parameters
lowcut = st.sidebar.number_input("Low Cutoff Frequency (Hz)", min_value=0.1, value=0.5, step=0.1)
highcut = st.sidebar.number_input("High Cutoff Frequency (Hz)", min_value=1.0, value=8.0, step=0.5)

# Patient parameters
st.sidebar.subheader("Patient Parameters")
weight = st.sidebar.number_input("Weight (kg)", min_value=1.0, value=90.0, step=1.0)
height_cm = st.sidebar.number_input("Height (cm)", min_value=1.0, value=183.0, step=1.0)


#### ------------------------- convert counts into ohms ------------------------------------------- ###
adc_bits = 20             # ADC resolution
vref = 1.0                # ADC reference voltage in Volts
gain = 10.0               # receive channel gain
i_exc = 0.0012            # # excitation current in Amps RMS (1.2 mA = 0.0012 A)

adc_full = (2**adc_bits - 1)
ohms_per_count = vref / (adc_full * gain * i_exc)

##----------------Function to read txt-------------------------------------------------------------##
def txt_to_dataframe(txt_content):
    try:
        # Try different separators
        for separator in [',', '\t', ' ', ';']:
            try:
                df = pd.read_csv(io.StringIO(txt_content), sep=separator, header=None, engine='python')
                if df.shape[1] > 1:  # If we have multiple columns
                    st.success(f"Successfully read TXT file with '{separator}' separator")
                    return df
            except:
                continue
        
        try:
            df = pd.read_fwf(io.StringIO(txt_content), header=None)
            if df.shape[1] > 1:
                st.success("Successfully read TXT file as fixed width")
                return df
        except:
            pass
            
        # If all else fails, return None
        st.error("Could not parse TXT file. Please check the format.")
        return None
        
    except Exception as e:
        st.error(f"Error reading TXT file: {str(e)}")
        return None


##--------------------MAIN CONTENT AREA------------------------------------------------------------------------------------###
if uploaded_file is not None:
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
 ##########----------for csv file----------------------------------------------------###########       
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file, header=0)
        
            # Display original data info
            st.subheader("Original Data")
            st.write(f"Data shape: {df.shape}")
            st.dataframe(df.head())
        
            processed_df = df.copy()
        
        # Display processed data
            st.subheader("Processed Data")
            st.write(f"Processed data shape: {processed_df.shape}")
            st.dataframe(processed_df.tail())
        
        # Column selection
            st.divider()
            st.divider()
            st.subheader("Column Selection")
        
        # Get all column names for selection
            all_columns = processed_df.columns.tolist()
        
        # ICG column selection (Z_ohm)
            st.write("Select the ICG column (Z_ohm):")
            icg_column = st.selectbox("ICG Column", options=all_columns, index=13 if len(all_columns) > 13 else 0)
        
        # Time range selection
            total_time = len(processed_df) / fs
            st.write(f"Total recording time: {total_time:.2f} seconds")
        
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.number_input("Start Time (s)", min_value=0.0, max_value=total_time, value=10.0, step=1.0)
            with col2:
                end_time = st.number_input("End Time (s)", min_value=start_time, max_value=total_time, value=total_time, step=1.0)
        
        # Extract Z_ohm data
            z_ohm = processed_df[icg_column]
            z_ohm = z_ohm * ohms_per_count

        # Signal processing functions
            def butter_bandpass(lowcut, highcut, fs, order=4):
                nyq = 0.5 * fs
                low = lowcut / nyq
                high = highcut / nyq
                b, a = butter(order, [low, high], btype='band')
                return b, a
        
            def bandpass_filter(data, lowcut, highcut, fs, order=4):
                b, a = butter_bandpass(lowcut, highcut, fs, order=order)
                y = filtfilt(b, a, data)
                return y
        
            def lowpass_baseline(sig, fs, cutoff_hz=0.5, order=4):
                nyq = 0.5 * fs
                b, a = butter(order, cutoff_hz/nyq, btype='low')
                return filtfilt(b, a, sig)
          
        # Create time array
            t = np.arange(len(z_ohm)) / fs
            z_ohm = -1 * z_ohm
            icg_filtered = bandpass_filter(z_ohm, lowcut, highcut, fs)
        
        
        # Zoom in on selected time range
            mask = (t >= start_time) & (t <= end_time)
            t_zoom = t[mask]
            z_zoom = icg_filtered[mask]
            raw_zoom = z_ohm[mask]
            dz_dt_zoom = np.gradient(z_zoom, 1/fs)
            dz_dt_smooth = bandpass_filter(dz_dt_zoom, lowcut, highcut, fs)

            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)
            t_seg = t[start_idx:end_idx]
            dz_seg = dz_dt_smooth[start_idx:end_idx]

            
        # Plotting
            st.divider()
            st.subheader("Signal Visualization")
        
        # Create tabs for different plots
            tab1, tab2, tab3 = st.tabs(["Raw ICG Signal", "Filtered ICG Signal", "First Derivative Signal"])
        
            with tab1:
                st.write(f"Raw ICG Signal (Zoom: {start_time}s - {end_time}s)")
                fig1, ax1 = plt.subplots(figsize=(24, 6))
                ax1.plot(t_zoom, raw_zoom, color="steelblue", linewidth=1)
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Impedance (Ohms)")
                ax1.set_title(f"Raw ICG Waveform ({start_time}s - {end_time}s)")
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
        
            with tab2:
                st.write(f"Filtered ICG Signal (Zoom: {start_time}s - {end_time}s)")
                fig2, ax2 = plt.subplots(figsize=(24, 6))
                ax2.plot(t_zoom, z_zoom, color="crimson", linewidth=1)
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Impedance (Ohms)")
                ax2.set_title(f"Filtered ICG Waveform ({start_time}s - {end_time}s)")
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
            
            with tab3:
                st.write(f"Comparison: Raw vs Filtered ICG Signal (Zoom: {start_time}s - {end_time}s)")
                fig3, ax3 = plt.subplots(figsize=(24, 6))
                ax3.plot(t_zoom, dz_dt_zoom, color="steelblue", linewidth=1, label="Raw Signal", alpha=0.7)
                ax3.plot(t_zoom, dz_dt_zoom, color="crimson", linewidth=1, label="Filtered Signal")
                ax3.set_xlabel("Time (s)")
                ax3.set_ylabel("Impedance (Ohms)")
                ax3.set_title(f"First Derivative ICG Waveform ({start_time}s - {end_time}s)")
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                st.pyplot(fig3)
        
            # Additional information
            st.divider()
            st.subheader("Processing Information")
            st.write(f"""
            - **Sampling Frequency**: {fs} Hz
            - **Bandpass Filter**: {lowcut} - {highcut} Hz
            - **Selected ICG Column**: {icg_column}
            - **Time Range**: {start_time} - {end_time} seconds
            - **Total Samples**: {len(z_ohm)}
            - **Zoom Samples**: {len(t_zoom)}
            """)
 ###########-------------------for txt file----------------------------------------------##########
        else:
        # Read the uploaded file
            txt_content = uploaded_file.getvalue().decode("utf-8")
        
        # Convert TXT to DataFrame
            df = txt_to_dataframe(txt_content)
        
        # Display original data info
            st.subheader("Original Data")
            st.write(f"Data shape: {df.shape}")
            st.dataframe(df.head())
        
        # Process ICG data
            def process_icg_file(input_df):
                # 1. Drop columns A (FA) and O (last col)
                processed_df = input_df.drop(input_df.columns[[0, -1]], axis=1)
            
            # 2. Convert HEX → DEC for columns B–M (now index 0–11 after dropping)
                for col in processed_df.columns[0:-1]:
                    processed_df[col] = processed_df[col].apply(lambda x: int(str(x), 16) if isinstance(x, str) else x)

                processed_df["PLOT-1"] = (processed_df.iloc[:, 5] * 65535) + (processed_df.iloc[:, 6] * 256) + processed_df.iloc[:, 7]
                processed_df["PLOT-2"] = (processed_df.iloc[:, 3] * 256) + processed_df.iloc[:, 4]
            
                return processed_df
        
            processed_df = process_icg_file(df)
        
            # Display processed data
            st.subheader("Processed Data")
            st.write(f"Processed data shape: {processed_df.shape}")
            st.dataframe(processed_df.head())
            
            # Column selection
            st.divider()
            st.divider()
            st.subheader("Column Selection")
            
            # Get all column names for selection
            all_columns = processed_df.columns.tolist()
            
            # ICG column selection (Z_ohm)
            st.write("Select the ICG column (Z_ohm):")
            icg_column = st.selectbox("ICG Column", options=all_columns, index=13 if len(all_columns) > 13 else 0)
            
            # Time range selection
            total_time = len(processed_df) / fs
            st.write(f"Total recording time: {total_time:.2f} seconds")
            
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.number_input("Start Time (s)", min_value=0.0, max_value=total_time, value=10.0, step=1.0)
            with col2:
                end_time = st.number_input("End Time (s)", min_value=start_time, max_value=total_time, value= total_time, step=1.0)
            
            # Extract Z_ohm data
            z_ohm = processed_df[icg_column]
            
            
            # Signal processing functions
            def butter_bandpass(lowcut, highcut, fs, order=4):
                nyq = 0.5 * fs
                low = lowcut / nyq
                high = highcut / nyq
                b, a = butter(order, [low, high], btype='band')
                return b, a
            
            def bandpass_filter(data, lowcut, highcut, fs, order=4):
                b, a = butter_bandpass(lowcut, highcut, fs, order=order)
                y = filtfilt(b, a, data)
                return y
            
            def lowpass_baseline(sig, fs, cutoff_hz=0.5, order=4):
                nyq = 0.5 * fs
                b, a = butter(order, cutoff_hz/nyq, btype='low')
                return filtfilt(b, a, sig)
                      
            
            # Create time array
            t = np.arange(len(z_ohm)) / fs
            z_ohm = -1 * z_ohm
            # Bandpass filter
            icg_filtered = bandpass_filter(z_ohm, lowcut, highcut, fs)
            
            # Zoom in on selected time range
            mask = (t >= start_time) & (t <= end_time)
            t_zoom = t[mask]
            z_zoom = icg_filtered[mask]
            raw_zoom = z_ohm[mask]
            dz_dt = np.gradient(z_ohm, 1/fs)
            dz_dt_zoom = np.gradient(z_zoom, 1/fs)
            dz_dt_smooth = bandpass_filter(dz_dt_zoom, lowcut, highcut, fs)
            
            # Plotting
            st.divider()
            st.subheader("Signal Visualization")
            
            # Create tabs for different plots
            tab1, tab2, tab3 = st.tabs(["Raw ICG Signal", "Filtered ICG Signal", "First Derivative signal"])
            
            with tab1:
                st.write(f"Raw ICG Signal (Interactive Zoom)")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=t, y=z_ohm, mode='lines', name='Full Raw Signal', 
                                            line=dict(color='lightgreen', width=1), opacity=0.7))
                fig1.add_trace(go.Scatter(x=t_zoom, y=raw_zoom, mode='lines', name='Zoomed Raw Signal', 
                                            line=dict(color='steelblue', width=3)))
                fig1.add_vrect(x0=start_time, x1=end_time, 
                                  fillcolor="lightgray", opacity=0.3, line_width=0,
                                  annotation_text="Zoom Area", annotation_position="top left")
                fig1.update_layout(
                        title="Raw ICG Waveform - Click and drag to zoom, double-click to reset",
                        xaxis_title="Time (s)",
                        yaxis_title="Impedance (Ohms)",
                        template="plotly_white",
                        height=600,
                        showlegend=True
                    )
                fig1.update_xaxes(rangeslider=dict(visible=True))
                st.plotly_chart(fig1, use_container_width=True)
            
            with tab2:
                st.write(f"Filtered ICG Signal (Interactive Zoom)")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=t, y=z_zoom, mode='lines', name='Full Filtered Signal', 
                                            line=dict(color='lightgreen', width=1), opacity=0.7))
                fig2.add_trace(go.Scatter(x=t_zoom, y=z_ohm, mode='lines', name='Zoomed Filtered Signal', 
                                            line=dict(color='cyan', width=3)))
                fig2.add_vrect(x0=start_time, x1=end_time, 
                                  fillcolor="lightgray", opacity=0.3, line_width=0,
                                  annotation_text="Zoom Area", annotation_position="top left")
                fig2.update_layout(
                        title="Filtered ICG Waveform - Click and drag to zoom, double-click to reset",
                        xaxis_title="Time (s)",
                        yaxis_title="Impedance (Ohms)",
                        template="plotly_white",
                        height=600,
                        showlegend=True
                    )
                fig2.update_xaxes(rangeslider=dict(visible=True))
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab3:
                st.write(f"First Derivative ICG Signal (Interactive Zoom)")
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=t, y=dz_dt, mode='lines', name='Full First derivative Signal', 
                                            line=dict(color='red',width=1), opacity=0.7))
                fig3.add_trace(go.Scatter(x=t_zoom, y=dz_dt_zoom, mode='lines', name='Zoomed First derivative Signal', 
                                            line=dict(color='lightgreen', width=3)))
                fig3.add_vrect(x0=start_time, x1=end_time, 
                                  fillcolor="lightgray", opacity=0.3, line_width=0,
                                  annotation_text="Zoom Area", annotation_position="top left")
                fig3.update_layout(
                        title="First Derivative Waveform - Click and drag to zoom, double-click to reset",
                        xaxis_title="Time (s)",
                        yaxis_title="Impedance (Ohms)",
                        template="plotly_white",
                        height=600,
                        showlegend=True
                    )
                fig3.update_xaxes(rangeslider=dict(visible=True))
                st.plotly_chart(fig3, use_container_width=True)


            st.divider()    
            st.subheader("Processing Information")
            st.write(f"""
            - **Sampling Frequency**: {fs} Hz
            - **Bandpass Filter**: {lowcut} - {highcut} Hz
            - **Selected ICG Column**: {icg_column}
            - **Time Range**: {start_time} - {end_time} seconds
            - **Total Samples**: {len(z_ohm)}
            - **Zoom Samples**: {len(t_zoom)}
            """)

###-----------------------------------------------------SIGNAL PROCESSING CALCULATIONS---------------------------------------------------------####
        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)
        t_seg = t[start_idx:end_idx]
        dz_seg = dz_dt_smooth[start_idx:end_idx]

                    # ------------------------
        # Detect C points (main peaks)
        # ------------------------
        c_peaks, _ = find_peaks(
            dz_seg,
            distance=fs*0.60,
            prominence=np.std(dz_seg)*0.1
        )

        # ------------------------ 
        # Detect B points (start of rise before C)
        # ------------------------
        b_points = []
        for c in c_peaks:
            search_region = dz_seg[max(0, c - int(1*fs)):c]
            b_rel = np.where(np.diff(np.sign(search_region)) > 0)[0]
            if len(b_rel) > 0:
                b_points.append(c - len(search_region) + b_rel[-1])
            else:
                b_points.append(c - 5)

        # ------------------------
        # Detect X points (true end of ejection, skip notches)
        # ------------------------
        x_points = []
        for c in c_peaks:
            search_region = dz_seg[c:c + int(0.255*fs)]
            if len(search_region) > 0:
                c_amp = dz_seg[c]
                valid_indices = [i for i, val in enumerate(search_region) if val < 0.8 * c_amp]
                if valid_indices:
                    min_idx = valid_indices[np.argmin([search_region[i] for i in valid_indices])]
                    x_points.append(c + min_idx)
                else:
                    x_points.append(c + np.argmin(search_region))
            else:
                x_points.append(c + 5)

        d2z_d2t = np.gradient(dz_dt_zoom, 1/fs)
        d2z_d2t_smooth = bandpass_filter(d2z_d2t, lowcut, highcut, fs)
        d2z_seg = d2z_d2t_smooth[start_idx:end_idx]

        c_peaks1, _ = find_peaks(
            d2z_seg,
            distance=fs*0.60,
            prominence=np.std(dz_seg)*0.1
        )

        d2z_d2t_max_values_new = d2z_seg[c_peaks1]
        d2z_d2t_values = np.abs(d2z_d2t_max_values_new)

        baseline = lowpass_baseline(z_ohm, fs, cutoff_hz=0.5, order=4)
        z0 = np.mean(baseline)
        pulsatile = z_ohm - baseline


#----------------------------------------CALCULATING PARAMETERS------------------------------------------------################

        # calculate VET ------------------------------------------------- 1
        VET_values = (np.array(x_points) - np.array(b_points)) / fs
        vet_values = VET_values * 1000

        # Calculate BMI---------------------------------------------------------- 12
        height_m = height_cm / 100
        bmi = weight / height_m**2

        # calculate BSA ---------------------------------------------------------- 13
        bsa = np.sqrt((height_cm * weight) / 3600)

        # calculate RR interval ------------------------------------------ 2
        c_times = t_seg[c_peaks]
        rr_intervals = np.diff(c_times)

        # calculate Heart rate -------------------------------------------- 3
        heart_rates = 60 / rr_intervals
        avg_heart_rate = heart_rates.mean()
        RR = 60/avg_heart_rate

        # calculate dzdt --------------------------------------------------- 4
        dz_dt_max_values_new = dz_seg[c_peaks]
        dz_dt_values = np.abs(dz_dt_max_values_new) 
        dz_dt_values = dz_dt_values* 3
        cb_values = (np.array(c_peaks) - np.array(b_points))
        cb_values_new = (np.array(c_peaks) - np.array(b_points)) / fs

        # Calculate Flow corrected Time (FTc)-------------------------------- 5
        FTc = vet_values/np.sqrt(RR)

        # Calculate Thoracic Fluid content ---------------------------------- 6
        tfc = 1000 / baseline
        value = bmi/fs

        # Calculate Ejection Time Ratio ------------------------------------- 7
        ETR = vet_values / (RR * 10 )

        # Calculate Index of Contractility ----------------------------------- 8
        IC = dz_dt_values * fs  / z0 * 15

        # Calculate Acceleration Index ---------------------------------------- 9
        ACI = 100 * d2z_d2t_values / cb_values.mean()

        # Calculate Heather index --------------------------------------------- 10
        HI = 2 * dz_dt_values / cb_values_new.mean()

        # Calculate Velocity Index --------------------------------------------- 11
        VI = 1000 * dz_dt_values / z0

        # Calculate Stroke Volume ------------------------------------------------- 14
        k = (baseline * bmi * value)  / (fs * bsa * tfc)
        sv =  ( np.mean(k) * vet_values )/ z0**2 * np.mean(tfc) * bmi + avg_heart_rate

        # Calculate Cardiac Output ------------------------------------------------- 15
        co = sv * avg_heart_rate

        # Calculate Cardiac Index -------------------------------------------------- 16
        ci = co / bsa 

        # calculate Stroke Volume Index --------------------------------------------- 17
        si = sv / bsa

 

####------------------------------------- SHOWING CALCULATED PARAMETERS ----------------------------------------- ######
        
        # Display calculated parameters
        st.divider()
        st.subheader("Calculated Parameters")

        col1, col2, col3  = st.columns(3)
        with col1:
            st.metric("SV [ml/beat]",f"{sv.mean():.2f} ")
        with col2:
            st.metric("CO [L/min]", f"{co.mean()/1000:.2f} ")
        with col3:
            st.metric("HR [bpm]", f"{avg_heart_rate:.0f}")


        col1, col2, col3= st.columns(3)
        with col1:
            st.metric("BMI", f"{bmi:.2f}")
        with col2:
            st.metric("BSA m²", f"{bsa:.2f} ")
        with col3:
            st.metric("z0", f"{z0:.2f}")
            col1, col2, col3= st.columns(3)


        col1, col2, col3= st.columns(3)
        with col1:
            st.metric("VET [m/s]", f"{vet_values.mean():.2f}")
        with col2:
            st.metric("dZ/dt", f"{dz_dt_values.mean():.2f} ")
        with col3:
            st.metric("RR-int [s]", f"{RR:.2f}")


        col1, col2, col3= st.columns(3)
        with col1:
            st.metric("ETR (%)", f"{ETR.mean():.2f}")
        with col2:
            st.metric("FTc [ms]", f"{FTc.mean():.2f} ")
        with col3:
            st.metric("TFC [ohm⁻¹]", f"{RR:.2f}")


        col1, col2, col3= st.columns(3)
        with col1:
            st.metric("IC [s⁻¹]", f"{IC.mean():.2f}")
        with col2:
            st.metric("SI [mL/beat/m²]", f"{si.mean():.2f} ")
        with col3:
            st.metric("CI [mL/min/m²]", f"{ci.mean():.2f}")


        col1, col2, col3= st.columns(3)
        with col1:
            st.metric("HI [s³]", f"{ETR.mean():.2f}")
        with col2:
            st.metric("VI [s⁻¹]", f"{FTc.mean():.2f} ")
        with col3:
            st.metric("ACI [s⁻²]", f"{RR:.2f}")


        # Process signals
        st.divider()
    
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please make sure you've uploaded a valid ICG data file in the correct format.")

else:
    st.info("Please upload a CSV file to begin processing.")
    
    # Display example of expected format
    st.subheader("Expected Data Format")
    st.write("""
    The app expects a CSV file with hexadecimal values in multiple columns. 
    The processing will:
    1. Drop the first and last columns
    2. Convert hexadecimal values to decimal
    3. Calculate AB, CD, VW, and XY composite values
    4. Process the ICG signal for analysis
    """)

# styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)



