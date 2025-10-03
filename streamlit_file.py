import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
import io

# Set page configuration
st.set_page_config(page_title="ICG Signal Processor", layout="wide")

# Title and description
st.title("ICG Signal Processing App")
st.markdown("""
This app processes Impedance Cardiography (ICG) data and visualizes the filtered signals.
Upload your CSV file and adjust the parameters to analyze the ICG waveform.
""")

# Sidebar for file upload and parameters
st.sidebar.header("Upload and Parameters")

# File upload
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Parameters section
st.sidebar.subheader("Processing Parameters")

# Sampling frequency
fs = st.sidebar.number_input("Sampling Frequency (Hz)", min_value=1, value=64, step=1)

# Filter parameters
lowcut = st.sidebar.number_input("Low Cutoff Frequency (Hz)", min_value=0.1, value=0.8, step=0.1)
highcut = st.sidebar.number_input("High Cutoff Frequency (Hz)", min_value=1.0, value=15.0, step=0.5)

# Patient parameters
st.sidebar.subheader("Patient Parameters")
weight = st.sidebar.number_input("Weight (kg)", min_value=1.0, value=90.0, step=1.0)
height_cm = st.sidebar.number_input("Height (cm)", min_value=1.0, value=183.0, step=1.0)

# Main content area
if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file, header=0)
        
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
            
            processed_df["AB"] = (processed_df.iloc[:, 1] * 256) + processed_df.iloc[:, 2]
            processed_df["CD"] = (processed_df.iloc[:, 3] * 256) + processed_df.iloc[:, 4]
            
            # Formula 2
            processed_df["VW"] = (processed_df.iloc[:, 5] * 65535) + (processed_df.iloc[:, 6] * 256) + processed_df.iloc[:, 7]
            processed_df["XY"] = (processed_df.iloc[:, 8] * 65535) + (processed_df.iloc[:, 9] * 256) + processed_df.iloc[:, 10]
            
            return processed_df
        
        processed_df = process_icg_file(df)
        
        # Display processed data
        st.subheader("Processed Data")
        st.write(f"Processed data shape: {processed_df.shape}")
        st.dataframe(processed_df.head())
        
        # Column selection
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
            start_time = st.number_input("Start Time (s)", min_value=0.0, max_value=total_time, value=50.0, step=1.0)
        with col2:
            end_time = st.number_input("End Time (s)", min_value=start_time, max_value=total_time, value=55.0, step=1.0)
        
        # Extract Z_ohm data
        z_ohm = processed_df[icg_column]
        
        # Calculate BMI and BSA
        height_m = height_cm / 100
        bmi = weight / height_m**2
        bsa = np.sqrt((height_cm * weight) / 3600)
        
        # Display calculated parameters
        st.subheader("Calculated Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BMI", f"{bmi:.2f}")
        with col2:
            st.metric("BSA", f"{bsa:.2f} m²")
        with col3:
            st.metric("BMI/fs", f"{(bmi)/fs:.4f}")
        
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
        
        # Process signals
        st.subheader("Signal Processing")
        
        # Apply filters
        baseline = lowpass_baseline(z_ohm, fs, cutoff_hz=0.5, order=4)
        z0 = np.mean(baseline)
        pulsatile = z_ohm - baseline
        
        # Bandpass filter
        icg_filtered = bandpass_filter(z_ohm, lowcut, highcut, fs)
        
        # Display basal impedance
        st.metric("Basal Impedance (Z0)", f"{z0:.2f} Ω")
        
        # Create time array
        t = np.arange(len(z_ohm)) / fs
        
        # Zoom in on selected time range
        mask = (t >= start_time) & (t <= end_time)
        t_zoom = t[mask]
        z_zoom = icg_filtered[mask]
        raw_zoom = z_ohm[mask]
        dz_dt_zoom = np.gradient(raw_zoom, 1/fs)
        
        # Plotting
        st.subheader("Signal Visualization")
        
        # Create tabs for different plots
        tab1, tab2, tab3 = st.tabs(["Raw ICG Signal", "Filtered ICG Signal", "Comparison"])
        
        with tab1:
            st.write(f"Raw ICG Signal (Zoom: {start_time}s - {end_time}s)")
            fig1, ax1 = plt.subplots(figsize=(20, 6))
            ax1.plot(t_zoom, raw_zoom, color="steelblue", linewidth=1)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Impedance (Ohms)")
            ax1.set_title(f"Raw ICG Waveform ({start_time}s - {end_time}s)")
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
        
        with tab2:
            st.write(f"Filtered ICG Signal (Zoom: {start_time}s - {end_time}s)")
            fig2, ax2 = plt.subplots(figsize=(20, 6))
            ax2.plot(t_zoom, dz_dt_zoom, color="crimson", linewidth=1)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Impedance (Ohms)")
            ax2.set_title(f"Filtered ICG Waveform ({start_time}s - {end_time}s)")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
        
        with tab3:
            st.write(f"Comparison: Raw vs Filtered ICG Signal (Zoom: {start_time}s - {end_time}s)")
            fig3, ax3 = plt.subplots(figsize=(20, 6))
            ax3.plot(t_zoom, z_zoom, color="steelblue", linewidth=1, label="Raw Signal", alpha=0.7)
            ax3.plot(t_zoom, dz_dt_zoom, color="crimson", linewidth=1, label="Filtered Signal")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Impedance (Ohms)")
            ax3.set_title(f"Raw vs Filtered ICG Waveform ({start_time}s - {end_time}s)")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
        
        # Additional information
        st.subheader("Processing Information")
        st.write(f"""
        - **Sampling Frequency**: {fs} Hz
        - **Bandpass Filter**: {lowcut} - {highcut} Hz
        - **Selected ICG Column**: {icg_column}
        - **Time Range**: {start_time} - {end_time} seconds
        - **Total Samples**: {len(z_ohm)}
        - **Zoom Samples**: {len(t_zoom)}
        """)
        
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

# Add some styling
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