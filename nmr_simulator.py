import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq

def Signal_1(f_kHz, t, T_decay, initial_phase):
    omega = 2 * np.pi * f_kHz  # Frequency in KHz
    return (np.cos(omega * t + initial_phase) + 1j * np.sin(omega * t + initial_phase)) * np.exp(-t / T_decay)

def get_user_inputs():
    frequencies = []
    decay_times = []
    signal_times = []
    phase_shifts = []

    global_initial_phase = float(input("Please enter the initial phase shift for the signals (degrees): "))
    initial_phase_rad = (np.pi * global_initial_phase) / 180

    for i in range(4):
        frequency_input = input(f"Please enter frequency for signal {i + 1} (KHz units): ")
        try:
            frequency = float(frequency_input.strip())
            frequencies.append(frequency)
        except ValueError:
            print("Error: Please make sure the frequency is a number.")
            return [], [], [], []

        try:
            T_Decay_time = float(input(f"Please enter a decay time T_2 value for signal {i + 1} (Milliseconds Units): "))
            decay_times.append(T_Decay_time)
        except ValueError:
            print("Error: Decay time must be a numeric value.")
            return [], [], [], []

        try:
            Signal_Time = float(input(f"Please enter the total time of the Signal for signal {i + 1} (Milliseconds Units): "))
            signal_times.append(Signal_Time)
        except ValueError:
            print("Error: Signal time must be a numeric value.")
            return [], [], [], []

        phase_shifts.append(initial_phase_rad)

        if i < 3:  # Ask if they want to add another signal up to 4 times
            add_another = input("Do you want to add another signal? (yes/no): ").strip().lower()
            if add_another != 'yes':
                break

    return frequencies, decay_times, signal_times, phase_shifts

# Use the function to get user inputs
frequencies, decay_times, signal_times, phase_shifts = get_user_inputs()
if not frequencies or not decay_times or not signal_times or not phase_shifts:
    print("Invalid inputs provided. Exiting...")
    frequencies = [1.0]  # Default value
    decay_times = [1.0]  # Default value
    signal_times = [5.0]  # Default value
    phase_shifts = [0.0]  # Default value
else:
    print(f"Frequencies: {frequencies} KHz, Decay Times T_2: {decay_times} ms, Signal Times: {signal_times} ms, Phase Shifts: {phase_shifts} Radians")

# Start time at 5 ms
max_signal_time = max(signal_times)
max_frequency = max(frequencies)
sampling_rate = 5 * max_frequency   # Ensure sufficient sampling rate
t = np.linspace(0, max_signal_time , int(max_signal_time * sampling_rate), endpoint=False)  # Increase sampling points
summed_signal = np.zeros(len(t), dtype=np.complex128)

for frequency_kHz, T_Decay_time, Signal_Time, phase_shift in zip(frequencies, decay_times, signal_times, phase_shifts):
    signal = Signal_1(frequency_kHz, t, T_Decay_time, initial_phase=phase_shift)
    summed_signal += signal

# Apply exponential phase shift to account for the delay
freqs = fftshift(fftfreq(len(t), 1 / sampling_rate))  # Get the frequencies in KHz

# Ask the user if they want to add noise to the FID signal
add_noise = input("Do you want to add noise to the FID signal? (yes/no): ").strip().lower()
if add_noise == 'yes':
    noise_type = input("Please choose noise type (gaussian/uniform): ").strip().lower()
    try:
        noise_level = float(input("Please enter the noise level (e.g., 0.1 for 10% of the signal amplitude in num value): "))
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level * np.max(np.real(summed_signal)), len(t))
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_level * np.max(np.real(summed_signal)), noise_level * np.max(np.real(summed_signal)), len(t))
        else:
            print("Invalid noise type. No noise will be added.")
            noise = np.zeros(len(t))  # No noise added
        summed_signal += noise
    except ValueError:
        print("Error: Noise level must be a numeric value.")
        noise = np.zeros(len(t))  # In case of error, no noise will be added
else:
    noise = np.zeros(len(t))  # No noise added

sum_signal = summed_signal.real
sum_signal_imag = summed_signal.imag
plt.figure(figsize=(12, 8))
plt.plot(t, sum_signal, label='Real Part of Summed Signal', linestyle='-', color='black')
#plt.plot(t, sum_signal_imag, label='imag Part of Summed Signal', linestyle='-', color='green')

plt.title('FID')
plt.xlabel('Time (ms)')
plt.ylabel('Signal Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Frequency Spectrum
fxt = fft(summed_signal)
fxt_shiftt = fftshift(fxt)
fxt_shifted = fftshift(fxt)
freqs = fftshift(fftfreq(len(t), 1 / sampling_rate))  # Get the frequencies in KHz

# Plot the frequency spectrum
plt.figure(figsize=(12, 8))
plt.plot(freqs, fxt_shifted.real,   linestyle='-', color='blue',label ='Real Part')
#plt.plot(freqs, fxt_shifted.imag, linestyle='-', color='green',label = 'image part')

plt.title('FT OF FID  Real  part')
plt.xlabel('Frequency (KHz)')
plt.ylabel('Magnitude')

# Dynamically set the x-axis limits
if frequencies:
    min_freq = min(frequencies)
    max_freq = max(frequencies)
    padding = 0.5 * (max_freq - min_freq) if max_freq != min_freq else 0.5 * max_freq
    x_min = min_freq - padding
    x_max = max_freq + padding
    plt.xlim([x_min, x_max])

plt.grid(True)
plt.show()

# Ask the user if they want to add a window function
def plot_decaying_window_results(previous_signal, windowed_sum_signal, fxt_shifted, fxt_windowed_shifted, freqs, decay_constant):
    plt.figure(figsize=(12, 8))
    plt.plot(t, previous_signal.real, label='Real Part of Previous Signal', linestyle='-', color='black')
    plt.plot(t, windowed_sum_signal, label=f'Real Part of Signal with Decaying Exponent (Window Function) (decay_constant={decay_constant})', linestyle='-', color='red')
    plt.title('FID - With and Without Decaying Exponent (Window Function)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Signal Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 1, 1)
    plt.plot(freqs, fxt_shifted.real, linestyle='-', color='blue', label='Previous Signal')
    plt.plot(freqs, fxt_windowed_shifted.real, linestyle='-', color='red', label=f'With Decaying Exponent (decay_constant={decay_constant})')
    plt.title('FT OF FID - Previous and With Decaying Exponent (Window Function)')
    plt.xlabel('Frequency (KHz)')
    plt.ylabel('Magnitude')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(freqs, fxt_windowed_shifted.real, linestyle='-', color='green')
    plt.title('FT OF FID - With Decaying Exponent (Window Function) Only')
    plt.xlabel('Frequency (KHz)')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, bottom=0.1)  # Adjust the bottom margin
    plt.show()

use_window_function = input("Do you want to add a decaying exponential window function to the signals? (yes/no): ").strip().lower()
if use_window_function == 'yes':
    adjust_values = 'yes'
    while adjust_values == 'yes':
        try:
            decay_constant = float(input("Please enter the decay constant for the window function (ms): "))
            window_function = np.exp(-t * decay_constant)

            windowed_signal = summed_signal * window_function

            # Plot the signal with the window function (if any)
            windowed_sum_signal = windowed_signal.real

            # Frequency spectrum for the signal with the window function (if any)
            fxt_windowed = np.fft.fft(windowed_signal)
            fxt_windowed_shifted = fftshift(fxt_windowed)

            plot_decaying_window_results(sum_signal, windowed_sum_signal, fxt_shifted, fxt_windowed_shifted, freqs, decay_constant)

            adjust_values = input("Do you want to adjust the decay constant value? (yes/no): ").strip().lower()

        except ValueError:
            print("Error: Decay constant must be a numeric value.")
else:
    windowed_signal = summed_signal

# Update previous_signal after window function
previous_signal = windowed_signal
fxt_shifted = fftshift(np.fft.fft(previous_signal))

# Phase Fix Section
def plot_phase_fix_results(previous_signal, signal_fix_zero_order, fxt_shifted, fxt_signal_fix_zero_order_shifted, freqs, phase):
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(t, previous_signal.real, 'b-', label='Previous Signal Real part ')
    #plt.plot(t, previous_signal.imag, 'r--', label='Previous Signal Image part')
    plt.title('FID - Time Domain Real  part')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t, signal_fix_zero_order.real, 'b-', label='signal_fix_zero_order Real part ')
   # plt.plot(t,signal_fix_zero_order.imag, 'r--', label='signal_fix_zero_order part')
    plt.title('FID - Time Domain Real part')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 1, 1)
    plt.plot(freqs, fxt_shifted.real, linestyle='-', color='blue', label='Previous Signal Real Part')
  #  plt.plot(freqs, fxt_shifted.imag, linestyle='-', color='red', label=f'Previous Signal Imagine Part Part')
    plt.title('FT OF FID - Previous Signal')
    plt.xlabel('Frequency (KHz)')
    plt.ylabel('Magnitude')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(freqs, fxt_signal_fix_zero_order_shifted.real,linestyle='-', color='blue',label = 'Real Part' )
 #   plt.plot(freqs, fxt_signal_fix_zero_order_shifted.imag, linestyle='-', color='red',label = 'Imagine Part')
    plt.title('FT OF FID - With Zero-Order Phase Correction Only With Real part')
    plt.xlabel('Frequency (KHz)')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, bottom=0.1)  # Adjust the bottom margin
    plt.show()

phase_correction = input("Do you want to apply zero-order phase correction? (yes/no): ").strip().lower()
if phase_correction == 'yes':
    adjust_phase = 'yes'
    while adjust_phase == 'yes':
        def get_user_inputs_1():
            phase = float(input("Please enter phase for zero order phase fix  (Degree Units) value: "))
            return phase

        phase = get_user_inputs_1()
        phase_shift = np.cos((-phase * (np.pi) / 180)) + 1j * np.sin((-phase * (np.pi) / 180))

        signal_to_fix = previous_signal

        signal_fix_zero_order = signal_to_fix * phase_shift

        # Perform FT
        fxt_signal_fix_zero_order = fft(signal_fix_zero_order)
        fxt_signal_fix_zero_order_shifted = fftshift(fxt_signal_fix_zero_order)

        plot_phase_fix_results(previous_signal, signal_fix_zero_order, fxt_shifted, fxt_signal_fix_zero_order_shifted, freqs, phase)

        adjust_phase = input("Do you want to adjust the phase value? (yes/no): ").strip().lower()

    # Update previous_signal after phase correction
    previous_signal = signal_fix_zero_order
    fxt_shifted = fftshift(fxt_signal_fix_zero_order)
else:
    fxt_signal_fix_zero_order_shifted = fxt_shifted
    signal_fix_zero_order = previous_signal

# Part 3 Lorentzian to Gaussian
def plot_lorentzian_gaussian_results(previous_signal, windowed_sum_signal_lorentz, fxt_shifted, fxt_windowed_lorentz_shifted, freqs, a, b):
    plt.figure(figsize=(12, 8))
    plt.plot(t, previous_signal.real, label='Real Part of Previous Signal', linestyle='-', color='black')
    plt.plot(t, windowed_sum_signal_lorentz, label=f'Real Part of Signal with Lorentzian to Gaussian (Window Function) (a={a}, b={b})', linestyle='-', color='red')
    plt.title('FID - With and Without Lorentzian to Gaussian Window Function')
    plt.xlabel('Time (ms)')
    plt.ylabel('Signal Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 1, 1)
    plt.plot(freqs, fxt_shifted.real, linestyle='-', color='blue', label='Previous Signal')
    plt.plot(freqs, fxt_windowed_lorentz_shifted.real, linestyle='-', color='red', label=f'With Lorentzian to Gaussian (a={a}, b={b})')
    plt.title('FT OF FID - Previous and With Lorentzian to Gaussian (Window Function)')
    plt.xlabel('Frequency (KHz)')
    plt.ylabel('Magnitude')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(freqs, fxt_windowed_lorentz_shifted.real, linestyle='-', color='green')
    plt.title('FT OF FID - With Lorentzian to Gaussian (Window Function) Only')
    plt.xlabel('Frequency (KHz)')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, bottom=0.1)  # Adjust the bottom margin
    plt.show()

use_lorentz_to_gauss = input("Do you want a transformation from Lorentzian to Gaussian (window function)? (yes/no): ").strip().lower()
if use_lorentz_to_gauss == 'yes':
    adjust_values_lorentz = 'yes'
    while adjust_values_lorentz == 'yes':
        try:
            user_input = input('Please enter values for a and b for exp(a*t - b*(t**2)), separated by a space: ')
            a, b = map(float, user_input.split())
            window_function = np.exp(t * a - b * (t**2))
            windowed_signal_lorentz = previous_signal * window_function

            # Plot the signal with the window function (if any)
            windowed_sum_signal_lorentz = windowed_signal_lorentz.real

            # Frequency spectrum for the signal with the window function (if any)
            fxt_windowed_lorentz = np.fft.fft(windowed_signal_lorentz)
            fxt_windowed_lorentz_shifted = fftshift(fxt_windowed_lorentz)

            plot_lorentzian_gaussian_results(previous_signal, windowed_sum_signal_lorentz, fxt_shifted, fxt_windowed_lorentz_shifted, freqs, a, b)

            adjust_values_lorentz = input("Do you want to adjust the values for a and b? (yes/no): ").strip().lower()

        except ValueError:
            print("Error: Please enter two numeric values separated by a space.")
    # Update previous_signal after Lorentzian to Gaussian
    previous_signal = windowed_signal_lorentz
    fxt_shifted = fftshift(np.fft.fft(previous_signal))
else:
    windowed_signal_lorentz = previous_signal
    # Update previous_signal after Lorentzian to Gaussian
    previous_signal = windowed_signal_lorentz

# Frequency spectrum for the signal with the chosen window function (if any)
fxt_windowed = np.fft.fft(previous_signal)
fxt_windowed_shifted = fftshift(fxt_windowed)

plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)
plt.plot(freqs, fxt_shifted.real, linestyle='-', color='blue')
plt.title('FT OF FID - Previous Signal')
plt.xlabel('Frequency (KHz)')
plt.ylabel('Magnitude')

plt.subplot(2, 1, 2)
plt.plot(freqs, fxt_windowed_shifted.real, linestyle='-', color='red', label='With Window Function')
plt.title('FT OF FID - With Window Function')
plt.xlabel('Frequency (KHz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.subplots_adjust(hspace=0.5, bottom=0.1)  # Adjust the bottom margin
plt.show()

# Final plot showing the initial and final signals in separate subplots for time and frequency domains
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.plot(t, summed_signal.real, linestyle='-', color='black')
plt.title('Initial Signal - Time Domain')
plt.xlabel('Time (ms)')
plt.ylabel('Signal Amplitude')

plt.subplot(2, 2, 2)
plt.plot(freqs, fxt_shiftt.real, linestyle='-', color='blue')
plt.title('Initial Signal - Frequency Domain')
plt.xlabel('Frequency (KHz)')
plt.ylabel('Magnitude')

plt.subplot(2, 2, 3)
plt.plot(t, previous_signal.real, linestyle='-', color='black')
plt.title('Final Signal - Time Domain')
plt.xlabel('Time (ms)')
plt.ylabel('Signal Amplitude')

plt.subplot(2, 2, 4)
plt.plot(freqs, fxt_windowed_shifted.real, linestyle='-', color='red')
plt.title('Final Signal - Frequency Domain')
plt.xlabel('Frequency (KHz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.subplots_adjust(hspace=0.5, bottom=0.1)  # Adjust the bottom margin
plt.show()
