 % OFDM Communication Chain Simulation
close all; clc; clear all;

% ------------------ PARAMETERS ------------------
M = 16;          % QAM order 
OFDM_b_size = 512;         % number of subcarriers
Ncp = 16;          % Cyclic prefix length (can vary 8-32) avoid ISI( adds time to each symbol, copy the end at the start)
DC = 1
Ngleft = ceil(0.05*OFDM_b_size)
Ngright = Ngleft
Nguard = Ngleft+Ngright+DC     % Guard subcarriers + DC (18 left, 17 right, 1 DC) avoid Inter carrier interference( spectral leakage), dc signal avoided for hardware reasons 0Hz
fs = 12.8e6;      % Sample rate
Nblocks = 200;         % Number of OFDM blocks
EbN0_dB = 0:2:20;      % SNR range
% QAM bits per symbol
k = log2(M);

% ------------------ GENERATE BITS & QAM SYMBOLS ------------------
Nsymbols = (OFDM_b_size - Nguard)*Nblocks;  % active subcarriers per OFDM * number of blocks
bits = randi([0 1], Nsymbols*k, 1); %Nbits=Nsymbol*k
bits_matrix = reshape(bits, k, []).';
symbols_index = bi2de(bits_matrix);
symbols = qammod(symbols_index, M, 'UnitAveragePower', true);

% ------------------ MAP SYMBOLS TO OFDM BLOCKS ------------------
% Initialize OFDM blocks (including zeros for guard subcarriers)
ofdm_blocks = zeros(OFDM_b_size, Nblocks);
active_idx = (Ngleft+1 : OFDM_b_size-Ngright)';  % exclude guard bands
active_idx = setdiff(active_idx, OFDM_b_size/2+1); % remove DC subcarrier
% Fill OFDM blocks
idx = 1;
for blk = 1:Nblocks
    ofdm_blocks(active_idx, blk) = symbols(idx:idx+length(active_idx)-1);
    idx = idx + length(active_idx);
end

% ------------------ OFDM TRANSMIT (IFFT + CP) ------------------
tx_time = [];
for blk = 1:Nblocks
    % IFFT
    tx_block = ifft(ifftshift(ofdm_blocks(:,blk)), OFDM_b_size);
    % Add cyclic prefix
    tx_block_cp = [tx_block(end-Ncp+1:end); tx_block];
    % Concatenate
    tx_time = [tx_time; tx_block_cp];
end

%Check orthogonality

% ------------------ RECEIVER (IDEAL CHANNEL) ------------------
rx_blocks = zeros(OFDM_b_size, Nblocks);

ptr = 1;
for blk = 1:Nblocks
    
    % Extract block including CP
    rx_block_cp = tx_time(ptr:ptr+OFDM_b_size+Ncp-1);
    
    % Remove CP
    rx_block = rx_block_cp(Ncp+1:end);
    
    % FFT
    rx_blocks(:,blk) = fftshift(fft(rx_block, OFDM_b_size));
    
    ptr = ptr + OFDM_b_size + Ncp;
end

error = norm(rx_blocks - ofdm_blocks);

disp(error)

%  1.0669e-14 Perfect !!!

% ------------------ PSD Plot ------------------
Nfft_psd = 2^nextpow2(length(tx_time));
X = fftshift(fft(tx_time, Nfft_psd));
PSD = abs(X).^2 / (Nfft_psd * fs);
f_psd = (-Nfft_psd/2 : Nfft_psd/2-1) * (fs/Nfft_psd);
figure;
plot(f_psd/1e6, 10*log10(PSD), 'LineWidth', 1.2);
xlabel('Frequency [MHz]'); ylabel('PSD [dB/Hz]');
title(sprintf('OFDM Transmit Signal PSD — %d-QAM', M));
grid on;
saveas(gcf,"PSDOFDM.pdf")
% ------------------ BER SIMULATION ------------------
BER_awgn = zeros(size(EbN0_dB));
BER_rayleigh = zeros(size(EbN0_dB));
BER_eq = zeros(size(EbN0_dB));

% Pre‑compute energy per bit (after normalisation)
total_bits = Nsymbols * k;
Eb = sum(abs(tx_time).^2) / total_bits;   % total energy / total bits

for i = 1:length(EbN0_dB)
    EbN0_lin = 10^(EbN0_dB(i)/10);
    N0 = Eb / EbN0_lin;          % one‑sided noise PSD
    sigma = sqrt(N0/2);          % std dev of real/imag noise components

    % =====================
    % 1️⃣ AWGN CHANNEL
    % =====================
    noise = sigma * (randn(size(tx_time)) + 1j*randn(size(tx_time)));
    rx_time_awgn = tx_time + noise;

    rx_symbols_awgn = [];
    idx = 1;
    for blk = 1:Nblocks
        rx_block_cp = rx_time_awgn(idx:idx+OFDM_b_size+Ncp-1);
        rx_block = rx_block_cp(Ncp+1:end);
        Y = fftshift(fft(rx_block, OFDM_b_size));
        rx_symbols_awgn = [rx_symbols_awgn; Y(active_idx)];
        idx = idx + OFDM_b_size + Ncp;
    end
    demapped = qamdemod(rx_symbols_awgn, M, 'UnitAveragePower', true);
    bits_hat = de2bi(demapped, k).';
    bits_hat = bits_hat(:);
    BER_awgn(i) = sum(bits ~= bits_hat) / length(bits);

    % =====================
    % 2️⃣ RAYLEIGH CHANNEL (NO EQ)
    % =====================
    tau = 3;                % delay of second path (samples)
    p = [1 0.5];            % path powers
    h = zeros(1, tau+1);
    h(1)   = sqrt(p(1)) * (randn + 1j*randn)/sqrt(2);
    h(end) = sqrt(p(2)) * (randn + 1j*randn)/sqrt(2);

    rx_time_ray = conv(tx_time, h);
    rx_time_ray = rx_time_ray(1:length(tx_time));   % truncate tail (CP absorbs IBI)
    noise = sigma * (randn(size(rx_time_ray)) + 1j*randn(size(rx_time_ray)));
    rx_time_ray = rx_time_ray + noise;

    rx_symbols_ray = [];
    idx = 1;
    for blk = 1:Nblocks
        rx_block_cp = rx_time_ray(idx:idx+OFDM_b_size+Ncp-1);
        rx_block = rx_block_cp(Ncp+1:end);
        Y = fftshift(fft(rx_block, OFDM_b_size));
        rx_symbols_ray = [rx_symbols_ray; Y(active_idx)];
        idx = idx + OFDM_b_size + Ncp;
    end
    demapped = qamdemod(rx_symbols_ray, M, 'UnitAveragePower', true);
    bits_hat = de2bi(demapped, k).';
    bits_hat = bits_hat(:);
    BER_rayleigh(i) = sum(bits ~= bits_hat) / length(bits);

    % =====================
    % 3️⃣ RAYLEIGH + ZERO‑FORCING EQUALIZATION
    % =====================
    H = fft(h, OFDM_b_size).';   % frequency response (natural order)

    rx_symbols_eq = [];
    idx = 1;
    for blk = 1:Nblocks
        rx_block_cp = rx_time_ray(idx:idx+OFDM_b_size+Ncp-1);
        rx_block = rx_block_cp(Ncp+1:end);
        Y = fft(rx_block, OFDM_b_size);        % natural order
        Yeq = Y ./ H;                           % ZF equalization
        Yeq_shifted = fftshift(Yeq);            % align with active_idx
        rx_symbols_eq = [rx_symbols_eq; Yeq_shifted(active_idx)];
        idx = idx + OFDM_b_size + Ncp;
    end
    demapped = qamdemod(rx_symbols_eq, M, 'UnitAveragePower', true);
    bits_hat = de2bi(demapped, k).';
    bits_hat = bits_hat(:);
    BER_eq(i) = sum(bits ~= bits_hat) / length(bits);
end

% ------------------ BER PLOT ------------------
figure;
semilogy(EbN0_dB, BER_awgn, 'o-', 'LineWidth', 2); hold on;
semilogy(EbN0_dB, BER_rayleigh, 's-', 'LineWidth', 2);
semilogy(EbN0_dB, BER_eq, '^-', 'LineWidth', 2);

% Theoretical AWGN reference for 16‑QAM
EbN0_lin_theory = 10.^(EbN0_dB/10);
SNR_per_bit = EbN0_lin_theory;
SNR_per_symbol = k * SNR_per_bit;
ber_theory = 4/k * (1 - 1/sqrt(M)) * ...
             qfunc(sqrt(3*SNR_per_symbol/(M-1)));
semilogy(EbN0_dB, ber_theory, 'k--', 'LineWidth', 1.5);

grid on; xlabel('E_b/N_0 [dB]'); ylabel('BER');
legend('AWGN (sim)', 'Rayleigh no EQ', 'Rayleigh + ZF EQ', ...
       'AWGN theory', 'Location', 'southwest');
title(sprintf('OFDM BER — %d-QAM with Two‑Path Rayleigh Channel', M));
ylim([1e-5 1]);
saveas(gcf,"BER_OFDM.pdf")
%CONSTELLATION VISUALIZATION
figure('Name', 'Constellation Comparison');
subplot(1,2,1);
plot(rx_symbols_ray(1:min(500,end)), '.');
title('Before EQ (Rayleigh)'); axis square; grid on;
subplot(1,2,2);
plot(rx_symbols_eq(1:min(500,end)), '.');
title('After ZF Equalization'); axis square; grid on;
saveas(gcf,"Constellation_OFDM.pdf")