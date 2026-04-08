% OFDM Communication Chain Simulation - Refined for Statistical Accuracy
close all; clc; clear all;

% ------------------ PARAMETERS ------------------
M = 16;                     % QAM order 
OFDM_b_size = 512;          % Number of subcarriers
Ncp = 16;                   % Cyclic prefix length
DC = 1;
Ngleft = ceil(0.05 * OFDM_b_size);
Ngright = Ngleft;
Nguard = Ngleft + Ngright + DC; 
fs = 12.8e6;                % Sample rate
Nblocks = 100;              % Blocks per trial
N_trials = 50;              % Number of channel realizations to average
EbN0_dB = 0:2:20;           % SNR range
k = log2(M);

% ------------------ GENERATE BITS & QAM SYMBOLS ------------------
Nsymbols_per_block = (OFDM_b_size - Nguard);
Ntotal_symbols = Nsymbols_per_block * Nblocks;
bits = randi([0 1], Ntotal_symbols * k, 1);
bits_matrix = reshape(bits, k, []).';
symbols_index = bi2de(bits_matrix);
symbols = qammod(symbols_index, M, 'UnitAveragePower', true);

% ------------------ MAP SYMBOLS TO OFDM BLOCKS ------------------
ofdm_blocks = zeros(OFDM_b_size, Nblocks);
active_idx = (Ngleft+1 : OFDM_b_size-Ngright)';  
active_idx = setdiff(active_idx, OFDM_b_size/2+1); % Remove DC

idx = 1;
for blk = 1:Nblocks
    ofdm_blocks(active_idx, blk) = symbols(idx:idx+Nsymbols_per_block-1);
    idx = idx + Nsymbols_per_block;
end

% ------------------ OFDM TRANSMIT (IFFT + CP) ------------------
tx_time = [];
for blk = 1:Nblocks
    tx_block = ifft(ifftshift(ofdm_blocks(:,blk)), OFDM_b_size);
    tx_block_cp = [tx_block(end-Ncp+1:end); tx_block];
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
saveas(gcf,"PSDOFDM_NCP16_QAM16.pdf")

% ------------------ BER SIMULATION ------------------
BER_awgn = zeros(size(EbN0_dB));
BER_rayleigh = zeros(size(EbN0_dB));
BER_eq_zf = zeros(size(EbN0_dB));

% Energy per bit calculation
Eb = sum(abs(tx_time).^2) / (length(bits));

for i = 1:length(EbN0_dB)
    EbN0_lin = 10^(EbN0_dB(i)/10);
    N0 = Eb / EbN0_lin;
    sigma = sqrt(N0/2);
    
    % --- 1. AWGN ONLY ---
    noise_awgn = sigma * (randn(size(tx_time)) + 1j*randn(size(tx_time)));
    rx_awgn = tx_time + noise_awgn;
    rx_syms_awgn = [];
    ptr = 1;
    for blk = 1:Nblocks
        block = rx_awgn(ptr+Ncp : ptr+Ncp+OFDM_b_size-1);
        Y = fftshift(fft(block, OFDM_b_size));
        rx_syms_awgn = [rx_syms_awgn; Y(active_idx)];
        ptr = ptr + OFDM_b_size + Ncp;
    end
    bits_awgn = reshape(de2bi(qamdemod(rx_syms_awgn, M, 'UnitAveragePower', true), k).', [], 1);
    BER_awgn(i) = sum(bits ~= bits_awgn) / length(bits);

% --- 2. RAYLEIGH (AVERAGING TRIALS) ---
    trial_ber_no_eq = zeros(N_trials, 1);
    trial_ber_zf = zeros(N_trials, 1);
    
    tau = 3; 
    p = [1 0.5]; 
    
    for t = 1:N_trials
        % New Channel Realization
        h = zeros(1, tau+1);
        h(1) = (randn + 1j*randn)/sqrt(2) * sqrt(p(1));
        h(end) = (randn + 1j*randn)/sqrt(2) * sqrt(p(2));
        H = fft(h, OFDM_b_size).';
        
        % Apply Channel + Noise
        rx_ray = conv(tx_time, h);
        rx_ray = rx_ray(1:length(tx_time)) + sigma*(randn(size(tx_time)) + 1j*randn(size(tx_time)));
        
        rx_syms_no_eq = [];
        rx_syms_zf = [];
        ptr = 1;
        for blk = 1:Nblocks
            block = rx_ray(ptr+Ncp : ptr+Ncp+OFDM_b_size-1);
            Y = fft(block, OFDM_b_size); % Natural order for EQ
            
            % No EQ
            Y_no_eq = fftshift(Y);
            rx_syms_no_eq = [rx_syms_no_eq; Y_no_eq(active_idx)];
            
            % Zero-Forcing EQ
            Y_zf = Y ./ H;
            Y_zf = fftshift(Y_zf);
            rx_syms_zf = [rx_syms_zf; Y_zf(active_idx)];
            
            ptr = ptr + OFDM_b_size + Ncp;
        end
        
        % De-map and count errors
        b_no_eq = reshape(de2bi(qamdemod(rx_syms_no_eq, M, 'UnitAveragePower', true), k).', [], 1);
        b_zf = reshape(de2bi(qamdemod(rx_syms_zf, M, 'UnitAveragePower', true), k).', [], 1);
        
        trial_ber_no_eq(t) = sum(bits ~= b_no_eq) / length(bits);
        trial_ber_zf(t) = sum(bits ~= b_zf) / length(bits);
    end
    
    BER_rayleigh(i) = mean(trial_ber_no_eq);
    BER_eq_zf(i) = mean(trial_ber_zf);
    
    fprintf('Processed EbN0 = %d dB\n', EbN0_dB(i));
end

% ------------------ BER PLOTTING ------------------
figure('Position', [100 100 800 600]);
semilogy(EbN0_dB, BER_awgn, 'bo-', 'LineWidth', 2, 'MarkerFaceColor', 'b'); hold on;
semilogy(EbN0_dB, BER_rayleigh, 'ro-', 'LineWidth', 2, 'MarkerFaceColor', 'r');
semilogy(EbN0_dB, BER_eq_zf, 'gd-', 'LineWidth', 2, 'MarkerFaceColor', 'g');

% Theoretical AWGN Reference
EbN0_lin_theory = 10.^(EbN0_dB/10);
ber_theory = 4/k * (1 - 1/sqrt(M)) * qfunc(sqrt(3*k*EbN0_lin_theory/(M-1)));
semilogy(EbN0_dB, ber_theory, 'k--', 'LineWidth', 1.5);

grid on;
xlabel('E_b/N_0 [dB]'); ylabel('BER');
legend('AWGN (sim)', 'Rayleigh (no EQ)', 'Rayleigh + ZF EQ', 'AWGN Theory');
title(sprintf('OFDM BER: %d-QAM, N_{cp}=%d', M, Ncp));
ylim([1e-5 1]);

% Fix sizing for PDF
set(gcf, 'PaperPositionMode', 'auto');
saveas(gcf, "BER_OFDM(CP=16)_QAM16_Blocksize2.pdf");

% ------------------ CONSTELLATION VISUALIZATION ------------------
% Plotting the results from the final EbN0 iteration (highest SNR)
figure('Name', 'Constellation Comparison', 'Position', [200 200 900 400]);

% Determine number of points safely
nPts_ray = min(1000, length(rx_syms_no_eq));
nPts_zf  = min(1000, length(rx_syms_zf));

% Plot Before Equalization
subplot(1,2,1);
plot(rx_syms_no_eq(1:nPts_ray), 'r.');
title('Before EQ (Rayleigh)'); 
xlabel('In-Phase'); ylabel('Quadrature');
axis square; grid on; xlim([-2 2]); ylim([-2 2]);

% Plot After Equalization
subplot(1,2,2);
plot(rx_syms_zf(1:nPts_zf), 'b.');
title('After ZF Equalization'); 
xlabel('In-Phase'); ylabel('Quadrature');
axis square; grid on; xlim([-2 2]); ylim([-2 2]);

% Export Constellation Figure
set(gcf, 'PaperPositionMode', 'auto');
saveas(gcf, "Constellation_OFDM(CP=16)_QAM16_block.pdf");