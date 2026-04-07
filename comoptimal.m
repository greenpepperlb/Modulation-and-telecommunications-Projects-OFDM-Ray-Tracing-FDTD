%% Step 1: RRC Communication Chain
close all; clc; clear all;

% Parameters
M   = 8;                    % QAM order — change freely (16, 64, 256...)
k   = log2(M);
Rs  = 5e6; 
fs  = 60e6;
sps = fs/Rs;                % = 12 samples per symbol
alpha    = 0.3;
RRC_span = 20;
Nbits    = floor(1e5/k)*k;
EbN0_dB  = 0:2:20;

% Bits & symbols
bits         = randi([0 1], Nbits, 1);
bits_matrix  = reshape(bits, k, []).';
symbol_index = bi2de(bits_matrix);
symbols      = qammod(symbol_index, M, 'UnitAveragePower', true); % Es = 1

% Upsample
symbols_up = upsample(symbols, sps);
Nsymbols   = length(symbols);

% ---------------------------------------------------------------
%  RRC from FREQUENCY DOMAIN (project requirement)
%  1) Build raised cosine H_RC(f) on a frequency grid
%  2) H_RRC(f) = sqrt(H_RC(f))
%  3) IFFT → time-domain taps
% ---------------------------------------------------------------
Nfilt  = RRC_span * sps + 1;              
T      = 1/Rs;
f_grid = (-floor(Nfilt/2):floor(Nfilt/2)) * (fs/Nfilt);

f1 = (1-alpha)/(2*T);   % lower roll-off edge
f2 = (1+alpha)/(2*T);   % upper roll-off edge

H_RC = zeros(1, Nfilt);
for idx = 1:Nfilt
    af = abs(f_grid(idx));
    if af <= f1
        H_RC(idx) = T;
    elseif af <= f2
        H_RC(idx) = (T/2)*(1 + cos(pi*T/alpha*(af - f1)));
    % else stays 0
    end
end

H_RRC = sqrt(H_RC);                                    % freq-domain RRC

g = real(fftshift(ifft(ifftshift(H_RRC))));            % time-domain taps

% Normalize so (g★g)[0] = 1  →  ISI-free unit gain at sample instants
g = g / sqrt(sum(g.^2));

% Transmit filtering
tx = conv(symbols_up, g);

% ---------------------------------------------------------------
%  BER loop — CORRECT noise formula
%
%  Filter normalization: sum(g²) = 1
%  ⟹ MF output signal   = symbol × 1  (unit gain)
%  ⟹ MF output noise/dim = σ² × sum(g²) = σ²
%  We want output noise/dim = N0/2  ⟹  σ = sqrt(N0/2)   ✓
%  (adding fs was ? )
% ---------------------------------------------------------------
BER    = zeros(size(EbN0_dB));
BER_th = zeros(size(EbN0_dB));
delay  = Nfilt - 1;                       % combined TX + RX filter delay

for i = 1:length(EbN0_dB)
    EbN0_lin = 10^(EbN0_dB(i)/10);
    Eb       = 1/k;                       % Es=1, so Eb = Es/k = 1/k
    N0       = Eb / EbN0_lin;
    sigma    = sqrt(N0/2);                % correct: one value per dimension

    noise       = sigma*(randn(size(tx)) + 1j*randn(size(tx)));
    rx          = tx + noise;
    rx_filtered = conv(rx, g);

    % Sample at symbol rate, skipping the filter delay
    rx_samples = rx_filtered(delay+1 : sps : delay + Nsymbols*sps);

    % Demap
    demapped        = qamdemod(rx_samples, M, 'UnitAveragePower', true);
    bits_hat_matrix = de2bi(demapped, k);
    bits_hat        = bits_hat_matrix.'; 
    bits_hat        = bits_hat(:);

    BER(i) = sum(bits ~= bits_hat) / Nbits;
    % Theoretical BER 
    BER_th(i) = berawgn(EbN0_dB(i), 'qam', M);
    
end

% After sampling rx_samples, compute noise at receiver
noise_at_output = rx_samples - symbols;

% Measure N0 per dimension (I and Q separately)
N0_meas_I = var(real(noise_at_output));
N0_meas_Q = var(imag(noise_at_output));
N0_measured = (N0_meas_I + N0_meas_Q) / 2;  % average both dims

% Compute actual Eb/N0 achieved
EbN0_measured_dB = 10*log10((1/k) / N0_measured);

% Compare with intended
fprintf('Intended  Eb/N0 = %.2f dB\n', EbN0_dB(i));
fprintf('Measured  Eb/N0 = %.2f dB\n', EbN0_measured_dB);
fprintf('N0 intended  = %.6f\n', N0);
fprintf('N0 measured  = %.6f\n', N0_measured);
% ---------------------------------------------------------------
%  Figure 1 — RRC filter shape
% ---------------------------------------------------------------
figure('Name','RRC Filter')
subplot(2,1,1)
plot(f_grid/1e6, H_RRC,'LineWidth',1.5)
xlabel('Frequency [MHz]'); ylabel('|H_{RRC}(f)|')
title('RRC — Frequency Domain'); grid on

subplot(2,1,2)
t_ax = (-(Nfilt-1)/2:(Nfilt-1)/2)/fs*1e6;
plot(t_ax, g,'LineWidth',1.5)
xlabel('Time [µs]'); ylabel('Amplitude')
title('RRC — Time Domain (via IFFT)'); grid on

% ---------------------------------------------------------------
%  Figure 2 — BER vs Eb/N0
% ---------------------------------------------------------------
figure('Name','BER')
semilogy(EbN0_dB, BER,'o-','LineWidth',2,'DisplayName','Simulated')
hold on
semilogy(EbN0_dB, BER_th,'s--','LineWidth',2,'DisplayName','Theoretical')
grid on
xlabel('E_b/N_0 [dB]'); ylabel('BER')
title(sprintf('BER vs E_b/N_0  —  %d-QAM', M))
legend; ylim([1e-5 1])
% ---------------------------------------------------------------
%  Figure 3 — PSD
% ---------------------------------------------------------------
Nfft  = 2^nextpow2(length(tx));
X     = fftshift(fft(tx, Nfft));
PSD   = abs(X).^2 / (Nfft*fs);
f_psd = linspace(-fs/2, fs/2, Nfft);

figure('Name','PSD')
plot(f_psd/1e6, 10*log10(PSD),'LineWidth',1.2)
xlabel('Frequency [MHz]'); ylabel('PSD [dB/Hz]')
title(sprintf('Transmit Signal PSD — %d-QAM', M)); grid on

% ---------------------------------------------------------------
%  Figure 4 — Constellation (high SNR so clusters are tight)
%  Works for any M automatically
% ---------------------------------------------------------------
sigma_c      = sqrt((1/k)/1000/2);              % Eb/N0 = 1000 (very high)
noise_c      = sigma_c*(randn(size(tx)) + 1j*randn(size(tx)));
rx_c         = conv(tx + noise_c, g);
rx_samp_c    = rx_c(delay+1 : sps : delay + Nsymbols*sps);
ideal_pts    = qammod(0:M-1, M, 'UnitAveragePower', true);

figure('Name','Constellation')
scatter(real(rx_samp_c), imag(rx_samp_c), 8, '.','MarkerEdgeAlpha',0.15,...
        'DisplayName','Received')
hold on
scatter(real(ideal_pts), imag(ideal_pts), 80, 'r','filled',...
        'DisplayName','Ideal points')
xlabel('I'); ylabel('Q')
title(sprintf('%d-QAM Constellation (high SNR)', M))
legend; grid on; axis equal