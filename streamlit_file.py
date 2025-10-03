import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="txt")

# Parameters section
st.sidebar.subheader("Processing Parameters")

# Sampling frequency
fs = st.sidebar.number_input("Sampling Frequency (Hz)", min_value=1, value=62, step=1)

# Filter parameters
lowcut = st.sidebar.number_input("Low Cutoff Frequency (Hz)", min_value=0.1, value=0.8, step=0.1)
highcut = st.sidebar.number_input("High Cutoff Frequency (Hz)", min_value=1.0, value=15.0, step=0.5)

# Patient parameters
st.sidebar.subheader("Patient Parameters")
weight = st.sidebar.number_input("Weight (kg)", min_value=1.0, value=90.0, step=1.0)
height_cm = st.sidebar.number_input("Height (cm)", min_value=1.0, value=183.0, step=1.0)


def txt_to_dataframe(txt_content):
    """
    Convert TXT file content to pandas DataFrame
    Supports space-separated, tab-separated, or comma-separated values
    """
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
        
        # If no separator works, try reading as fixed width
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

# Main content area
if uploaded_file is not None:
    try:
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
            end_time = st.number_input("End Time (s)", min_value=start_time, max_value=total_time, value=70.0, step=1.0)
        
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
        dz_dt = np.gradient(z_ohm, 1/fs)
        dz_dt_zoom = np.gradient(raw_zoom, 1/fs)
        
        # Plotting
        st.subheader("Signal Visualization")
        
        # Create tabs for different plots
        tab1, tab2, tab3 = st.tabs(["Raw ICG Signal", "Filtered ICG Signal", "First Derivative signal"])
        
        with tab1:
            st.write(f"Raw ICG Signal (Interactive Zoom)")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=t, y=z_ohm, mode='lines', name='Full Raw Signal', 
                                        line=dict(color='lightgray', width=1), opacity=0.7))
            fig1.add_trace(go.Scatter(x=t_zoom, y=raw_zoom, mode='lines', name='Zoomed Raw Signal', 
                                        line=dict(color='steelblue', width=3)))
            fig1.add_vrect(x0=start_time, x1=end_time, 
                              fillcolor="lightyellow", opacity=0.3, line_width=0,
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
                                        line=dict(color='lightgray', width=1), opacity=0.7))
            fig2.add_trace(go.Scatter(x=t_zoom, y=z_ohm, mode='lines', name='Zoomed Filtered Signal', 
                                        line=dict(color='crimson', width=3)))
            fig2.add_vrect(x0=start_time, x1=end_time, 
                              fillcolor="lightyellow", opacity=0.3, line_width=0,
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
                                        line=dict(color='lightgray', width=1), opacity=0.7))
            fig3.add_trace(go.Scatter(x=t_zoom, y=dz_dt_zoom, mode='lines', name='Zoomed First derivative Signal', 
                                        line=dict(color='crimson', width=3)))
            fig3.add_vrect(x0=start_time, x1=end_time, 
                              fillcolor="lightyellow", opacity=0.3, line_width=0,
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
