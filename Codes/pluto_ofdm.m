%% ELEC-H-401 - Pluto OFDM transmission/reception
% Real-time OFDM packet experiment for ADALM-Pluto.
% The script transmits a periodic OFDM packet, receives a burst, detects the
% packet by preamble cross-correlation, estimates CFO, estimates the channel,
% equalizes the OFDM data symbols, computes BER, and plots the constellation.

clear; clc; close all;
rng(42);

%% USER SETTINGS
centerFrequency = 1e9;      % Pluto RF center frequency [Hz]
Fs           = 12.8e6;   % Pluto baseband sample rate [Hz]
txGain          = -20;      % Pluto TX gain [dB], range about -90 to 0
rxGain          = 20;       % Pluto RX gain [dB], range about -4 to 71
samplesPerFrame = 200000;   % RX burst length
txScale         = 0.25;     % Keep the transmitted waveform away from clipping

N       = 512;              % OFDM FFT size
CP      = 32;               % Cyclic prefix length
Ng_L    = 18;               % Left guard subcarriers
Ng_R    = 17;               % Right guard subcarriers
M       = 16;               % QAM order
n_sym   = 20;               % Data OFDM symbols per packet
L_ch    = CP;               % Assumed max channel length for denoising

%% MODEM PARAMETERS
guard_idx = [1:Ng_L, N/2+1, N-Ng_R+1:N];
active = setdiff(1:N, guard_idx).';
Nsc = numel(active);
bits_per_sym = log2(M);

qam_mod = @(x) qammod(x, M, 'UnitAveragePower', true);
qam_demod = @(x) qamdemod(x, M, 'UnitAveragePower', true);

F_N = fft(eye(N));
F_sub = F_N(active, 1:L_ch);

%% TRANSMIT PACKET
% Use two identical preambles. The first pair supports coarse CFO estimation;
% the second preamble is then used for channel estimation.
preamble_syms = 2*randi([0 1], Nsc, 1) - 1;
preamble_ofdm = ofdm_tx(preamble_syms, N, CP, active);
ofdm_len = N + CP;

data_tx = randi([0 M-1], Nsc, n_sym);
data_syms = reshape(qam_mod(data_tx(:)), Nsc, n_sym);

tx_packet = [preamble_ofdm; preamble_ofdm];
for m_idx = 1:n_sym
    tx_packet = [tx_packet; ofdm_tx(data_syms(:,m_idx), N, CP, active)]; %#ok<AGROW>
end

tx_buffer = txScale * tx_packet / max(abs(tx_packet));

%% PLUTO TRANSMIT/RECEIVE
assert(exist('sdrtx', 'file') == 2 && exist('sdrrx', 'file') == 2, ...
    ['Pluto SDR functions were not found. Install the Communications Toolbox ' ...
     'Support Package for Analog Devices ADALM-Pluto Radio.']);

txPluto = sdrtx('Pluto', ...
    'RadioID', 'usb:0', ...
    'Gain', txGain, ...
    'CenterFrequency', centerFrequency, ...
    'BasebandSampleRate', Fs);

rxPluto = sdrrx('Pluto', ...
    'RadioID', 'usb:0', ...
    'CenterFrequency', centerFrequency, ...
    'GainSource', 'Manual', ...
    'Gain', rxGain, ...
    'BasebandSampleRate', Fs, ...
    'EnableBurstMode', true, ...
    'NumFramesInBurst', 1, ...
    'SamplesPerFrame', samplesPerFrame, ...
    'OutputDataType', 'double');

cleanupObj = onCleanup(@() release_radios(txPluto, rxPluto));
txPluto.transmitRepeat(tx_buffer);
pause(0.2);

[rx_capture, datavalid, overflow] = rxPluto();
if ~datavalid
    error('Pluto returned an invalid RX frame.');
end
if overflow
    warning('Samples were dropped during reception. Reduce Fs or samplesPerFrame.');
end

rx_capture = rx_capture(:);
rx_capture = rx_capture - mean(rx_capture);

%% PACKET DETECTION AND CFO ESTIMATION
search_preamble = [preamble_ofdm; preamble_ofdm];
[packet_start, metric] = estimate_packet_start(rx_capture, search_preamble);

packet_len = length(tx_packet);
if packet_start + packet_len - 1 > length(rx_capture)
    error('Detected packet does not fit in the captured frame. Increase samplesPerFrame.');
end

rx_packet = rx_capture(packet_start:packet_start + packet_len - 1);

rx_pr1 = rx_packet(1:ofdm_len);
rx_pr2 = rx_packet(ofdm_len+1:2*ofdm_len);
cfo_rad_per_sample = angle(sum(conj(rx_pr1) .* rx_pr2)) / ofdm_len;
n = (0:length(rx_packet)-1).';
rx_packet = rx_packet .* exp(-1j * cfo_rad_per_sample * n);
cfo_hz = cfo_rad_per_sample * Fs / (2*pi);

%% CHANNEL ESTIMATION
rx_pr2 = rx_packet(ofdm_len+1:2*ofdm_len);
Yp = ofdm_rx(rx_pr2, N, CP, active);
H_raw = Yp ./ preamble_syms;
h_ml = F_sub \ H_raw;
H_den = F_sub * h_ml;

%% DATA EQUALIZATION AND BER
data_rx = zeros(Nsc, n_sym);
eq_symbols = zeros(Nsc, n_sym);
n_decoded = 0;
data_start = 2 * ofdm_len + 1;

for m_idx = 1:n_sym
    idx_s = data_start + (m_idx-1)*ofdm_len;
    idx_e = idx_s + ofdm_len - 1;
    if idx_e > length(rx_packet)
        break;
    end

    Y = ofdm_rx(rx_packet(idx_s:idx_e), N, CP, active);
    eq_symbols(:,m_idx) = Y ./ H_den;
    data_rx(:,m_idx) = qam_demod(eq_symbols(:,m_idx));
    n_decoded = n_decoded + 1;
end

if n_decoded == 0
    error('No OFDM data symbols were decoded.');
end

tx_ref = data_tx(:,1:n_decoded);
rx_ref = data_rx(:,1:n_decoded);
ber = biterr(tx_ref(:), rx_ref(:), bits_per_sym) / (numel(tx_ref) * bits_per_sym);

fprintf('Detected packet start: %d samples\n', packet_start);
fprintf('Estimated CFO: %.2f Hz\n', cfo_hz);
fprintf('Decoded OFDM data symbols: %d / %d\n', n_decoded, n_sym);
fprintf('BER after synchronization and denoised channel equalization: %.4g\n', ber);

%% FIGURES
figure('Name','Pluto packet detection');
plot(metric, 'LineWidth', 1.2);
grid on;
xlabel('Candidate packet start (samples)');
ylabel('Correlation magnitude');
title('OFDM preamble packet detection');

figure('Name','Pluto synchronized constellation');
rx_plot = eq_symbols(:,1:n_decoded);
plot(real(rx_plot(:)), imag(rx_plot(:)), '.', 'MarkerSize', 6);
hold on;
ideal_constellation = qam_mod((0:M-1).');
plot(real(ideal_constellation), imag(ideal_constellation), 'kx', ...
    'LineWidth', 1.6, 'MarkerSize', 10);
grid on;
axis equal;
xlabel('In-phase');
ylabel('Quadrature');
title(sprintf('Pluto received constellation after sync, BER = %.3g', ber));
legend('Received symbols', 'Ideal 16-QAM points', 'Location', 'best');

%% LOCAL FUNCTIONS
function ofdm = ofdm_tx(syms, N, CP, active)
    X = zeros(N, 1);
    X(active) = syms;
    x = ifft(X, N) * sqrt(N);
    ofdm = [x(end-CP+1:end); x];
end

function syms = ofdm_rx(r, N, CP, active)
    x = r(CP+1:CP+N);
    X = fft(x, N) / sqrt(N);
    syms = X(active);
    syms = syms(:);
end

function [start_hat, metric] = estimate_packet_start(rx, preamble)
    metric = abs(conv(rx(:), flipud(conj(preamble(:))), 'valid'));
    [~, idx] = max(metric);
    start_hat = idx;
end

function release_radios(txPluto, rxPluto)
    try
        release(txPluto);
    catch
    end
    try
        release(rxPluto);
    catch
    end
end
