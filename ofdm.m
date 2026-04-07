%% OFDM Communication Chain Simulation
close all; clc; clear all;

%% ------------------ PARAMETERS ------------------
M = 16;          % QAM order (can be 4,16,64,256,1024)
OFDM_b_size = 512;         % OFDM block size (number of subcarriers)
Ncp = 16;          % Cyclic prefix length (can vary 8-32)
DC = 1
Ngleft = ceil(0.05*OFDM_b_size)
Ngright = Ngleft
Nguard = Ngleft+Ngright+DC     % Guard subcarriers + DC (18 left, 17 right, 1 DC)
fs = 12.8e6;      % Sample rate
Nblocks = 200;         % Number of OFDM blocks
EbN0_dB = 0:2:20;      % SNR range
% QAM bits per symbol
k = log2(M);

%% ------------------ GENERATE BITS & QAM SYMBOLS ------------------
Nsymbols = (OFDM_b_size - Nguard)*Nblocks;  % active subcarriers per OFDM * number of blocks
bits = randi([0 1], Nsymbols*k, 1); %Nbits=Nsymbol*k
bits_matrix = reshape(bits, k, []).';
symbols_index = bi2de(bits_matrix);
symbols = qammod(symbols_index, M, 'UnitAveragePower', true);

%% ------------------ MAP SYMBOLS TO OFDM BLOCKS ------------------
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

%% ------------------ OFDM TRANSMIT (IFFT + CP) ------------------
tx_time = [];
for blk = 1:Nblocks
    % IFFT
    tx_block = ifft(ifftshift(ofdm_blocks(:,blk)), OFDM_b_size);
    % Add cyclic prefix
    tx_block_cp = [tx_block(end-Ncp+1:end); tx_block];
    % Concatenate
    tx_time = [tx_time; tx_block_cp];
end

%% ------------------ POWER SPECTRAL DENSITY ------------------
Nfft_psd = 2^nextpow2(length(tx_time));
X = fftshift(fft(tx_time, Nfft_psd));
PSD = abs(X).^2/(Nfft_psd*fs);
f_psd = linspace(-fs/2, fs/2, Nfft_psd);

figure('Name','OFDM PSD')
plot(f_psd/1e6,10*log10(PSD),'LineWidth',1.2)
xlabel('Frequency [MHz]'); ylabel('PSD [dB/Hz]')
title(sprintf('OFDM Transmit Signal PSD — %d-QAM', M))
grid on

%% ------------------ BER SIMULATION — AWGN CHANNEL ------------------
BER = zeros(size(EbN0_dB));
BER_theo = zeros(size(EbN0_dB));

for i = 1:length(EbN0_dB)
    EbN0_lin = 10^(EbN0_dB(i)/10);
    Es = 1;                     % unit power
    N0 = Es/(k*EbN0_lin);
    sigma = sqrt(N0/2);
    
    noise = sigma*(randn(size(tx_time)) + 1j*randn(size(tx_time)));
    rx_time = tx_time + noise;
    
    % ------------------ OFDM RECEIVE ------------------
    rx_symbols = [];
    idx = 1;
    for blk = 1:Nblocks
        rx_block_cp = rx_time(idx:idx+OFDM_b_size+Ncp-1);
        % remove cyclic prefix
        rx_block = rx_block_cp(Ncp+1:end);
        % FFT
        Y = fftshift(fft(rx_block, OFDM_b_size));
        rx_symbols = [rx_symbols; Y(active_idx)];
        idx = idx + OFDM_b_size + Ncp;
    end
    
    % ------------------ DEMODULATION ------------------
    demapped = qamdemod(rx_symbols, M, 'UnitAveragePower', true);
    bits_hat_matrix = de2bi(demapped, k);
    bits_hat = bits_hat_matrix.';
    bits_hat = bits_hat(:);
    
    BER(i) = sum(bits ~= bits_hat)/length(bits);
    BER_theo(i) = berawgn(EbN0_dB(i), 'qam', M);
end

figure('Name','OFDM BER')
semilogy(EbN0_dB, BER,'o-','LineWidth',2,'DisplayName','Simulated')
hold on
semilogy(EbN0_dB, BER_theo,'s--','LineWidth',2,'DisplayName','Theoretical')
grid on
xlabel('E_b/N_0 [dB]'); ylabel('BER')
title(sprintf('OFDM BER vs Eb/N0 — %d-QAM', M))
legend; ylim([1e-5 1])

%% ------------------ TWO-PATH RAYLEIGH CHANNEL (OPTIONAL) ------------------
% Path delays in samples
tau = [0 3];          % first path 0, second path delayed by 3 samples
p   = [1 0.5];        % path powers
rx_time_chan = conv(tx_time, [sqrt(p(1)) sqrt(p(2))]);  % simple 2-tap channel
rx_time_chan = rx_time_chan(1:length(tx_time));          % truncate to same length
% Add AWGN
SNR_dB = 15;
rx_time_chan = rx_time_chan + 10^(-SNR_dB/20)*(randn(size(rx_time_chan)) + 1j*randn(size(rx_time_chan)));

% Frequency-domain equalization
H = fft([sqrt(p(1)) zeros(1,OFDM_b_size-2) sqrt(p(2))]);  % simple channel freq response
rx_blocks_eq = zeros(OFDM_b_size,length(rx_time)/ (OFDM_b_size+Ncp));
idx = 1;
for blk = 1:Nblocks
    rx_block_cp = rx_time_chan(idx:idx+OFDM_b_size+Ncp-1);
    rx_block = rx_block_cp(Ncp+1:end);
    Y = fftshift(fft(rx_block, OFDM_b_size));
    Y_eq = Y./H.';   % frequency-domain equalization
    rx_blocks_eq(:,blk) = Y_eq;
    idx = idx + OFDM_b_size + Ncp;
end

disp('OFDM simulation completed.')