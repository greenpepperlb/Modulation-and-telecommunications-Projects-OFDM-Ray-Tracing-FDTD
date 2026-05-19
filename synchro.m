%% ELEC-H-401 - Part 3: OFDM Synchronization
% Figures generated:
%   Fig1_Channel_Est_Error.pdf      BER impact of channel estimation error
%   Fig2_Cross_Correlation.pdf      packet timing correlation + RMSE
%   Fig3_Channel_Denoising.pdf      raw LS vs denoised CE MSE
%   Fig4_Unified_BER.pdf            final synchronized OFDM BER comparison
%   Fig5_Synchronized_Constellation.pdf received QAM constellation after sync
%   Fig6_Channel_Estimate_Example.pdf raw/denoised channel estimate example
%   Fig7_Cross_Correlation_Example.pdf packet timing correlation example
clear; clc; close all;
rng(42);

%% PARAMETERS
N       = 512;          % OFDM block size (FFT)
CP      = 32;           % Cyclic prefix length
Ng_L    = 18;           % Guard subcarriers left
Ng_R    = 17;           % Guard subcarriers right
M       = 16;           % QAM order
Rsamp   = 12.8e6;       % Sample rate [Hz]
Novs    = 4;            % Oversampling factor for cross-correlation
n_sym   = 10;           % Data OFDM symbols per packet

path_delay = 4;                         % second path delay [OFDM samples]
h = zeros(path_delay + 1, 1);
h(1) = 1;
h(path_delay + 1) = sqrt(0.5);          % two-path channel

guard_idx = [1:Ng_L, N/2+1, N-Ng_R+1:N];
active    = setdiff(1:N, guard_idx).';
Nsc       = length(active);

bits_per_sym = log2(M);
plot_floor   = 1e-5;

qam_mod   = @(x) qammod(x, M, 'UnitAveragePower', true);
qam_demod = @(x) qamdemod(x, M, 'UnitAveragePower', true);

H_true        = fft(h, N);
H_true_active = H_true(active);
H_true_active = H_true_active(:);

% Known OFDM preamble, reused by the synchronization algorithms.
preamble_syms = 2*randi([0 1], Nsc, 1) - 1;   % BPSK, unit power
preamble_ofdm = ofdm_tx(preamble_syms, N, CP, active);

L     = CP;
F_N   = fft(eye(N));
F_sub = F_N(active, 1:L);

%% STEP 1 - Impact of channel estimation error on BER
fprintf('=== Step 1: Channel estimation error impact ===\n');

err_std_vals = [0 0.02 0.05 0.10];
EbN0_dB_range = 0:2:20;
BER_est_err  = zeros(length(err_std_vals), length(EbN0_dB_range));
N_frames = 1500;

for k = 1:length(err_std_vals)
    err_std = err_std_vals(k);
    fprintf('  Running CSI error std = %.0f %%\n', 100*err_std);

    for i = 1:length(EbN0_dB_range)
        ebn0_lin = 10^(EbN0_dB_range(i)/10);
        total_errors = 0;
        total_bits = 0;

        for f = 1:N_frames
            data_tx = randi([0 M-1], Nsc, 1);
            syms_tx = qam_mod(data_tx);

            % Slide 25 is an AWGN-channel experiment: the only channel
            % impairment here is the artificial error on H_hat.
            noise_var = 1 / (bits_per_sym * ebn0_lin);
            syms_rx = syms_tx + sqrt(noise_var/2) * ...
                (randn(size(syms_tx)) + 1j*randn(size(syms_tx)));

            if err_std == 0
                H_est = ones(Nsc, 1);
            else
                est_err = err_std * (randn(Nsc,1) + 1j*randn(Nsc,1)) / sqrt(2);
                H_est = 1 + est_err;
            end

            data_rx = qam_demod(syms_rx ./ H_est);
            total_errors = total_errors + biterr(data_tx, data_rx, bits_per_sym);
            total_bits   = total_bits + Nsc * bits_per_sym;
        end

        BER_est_err(k, i) = total_errors / total_bits;
    end
end

fig1 = figure('Name','Fig1 - BER impact of channel estimation error', ...
    'Color', 'w', 'Units', 'pixels', 'Position', [100 100 900 650]);
hold on;
markers = {'+', 'o', '*', '<'};
legend_labels = {'0', '2%', '5%', '10%'};
for k = 1:length(err_std_vals)
    semilogy(EbN0_dB_range, max(BER_est_err(k,:), plot_floor), ...
        ['-' markers{k}], 'LineWidth', 1.0, 'Color', 'k', ...
        'MarkerSize', 8, 'DisplayName', legend_labels{k});
end
grid on;
grid minor;
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate (BER)');
title('Impact of channel estimation error on BER');
legend('Location','northeast');
ylim([plot_floor 1]);
xlim([EbN0_dB_range(1) EbN0_dB_range(end)]);
set(gca, 'YScale', 'log', 'FontSize', 12, 'LineWidth', 0.8, ...
    'TickDir', 'in', 'Box', 'on');
export_if_valid(fig1, 'Fig1_Channel_Est_Error.pdf');

%% STEP 2 - OFDM preamble cross-correlation timing on oversampled samples
fprintf('\n=== Step 2: Oversampled OFDM cross-correlation packet timing ===\n');

preamble_ovs = oversample_signal(preamble_ofdm, Novs);
timing_path_delays = 1:2:CP;
timing_snr_range = -30:5:15;
timing_selected_delays = unique([1 path_delay 8 16 CP]);
timing_fixed_snr = 0;
N_timing_trials = 200;
timing_lead_samples = 48;
timing_tail_samples = 96;

toa_rmse_delay = zeros(size(timing_path_delays));
toa_bias_delay = zeros(size(timing_path_delays));
toa_hit_delay = zeros(size(timing_path_delays));

fprintf('  Sweeping second-path delay at RX SNR = %d dB\n', timing_fixed_snr);
for d_idx = 1:length(timing_path_delays)
    delay_samp = timing_path_delays(d_idx);
    [toa_rmse_delay(d_idx), toa_bias_delay(d_idx), toa_hit_delay(d_idx)] = ...
        ofdm_toa_stats(preamble_ovs, Novs, delay_samp, timing_fixed_snr, ...
        N_timing_trials, timing_lead_samples, timing_tail_samples);
end

toa_rmse_snr = zeros(length(timing_snr_range), length(timing_selected_delays));
for d_idx = 1:length(timing_selected_delays)
    delay_samp = timing_selected_delays(d_idx);
    fprintf('  Sweeping RX SNR for second-path delay = %d samples\n', delay_samp);
    for s_idx = 1:length(timing_snr_range)
        toa_rmse_snr(s_idx,d_idx) = ofdm_toa_stats(preamble_ovs, Novs, ...
            delay_samp, timing_snr_range(s_idx), N_timing_trials, ...
            timing_lead_samples, timing_tail_samples);
    end
end

fig2 = figure('Name','Fig2 - Oversampled OFDM packet timing', ...
    'Color', 'w', 'Units', 'pixels', 'Position', [100 100 1000 420]);
subplot(1,2,1);
yyaxis left;
plot(timing_path_delays, toa_rmse_delay, '-o', 'LineWidth', 1.5, ...
    'DisplayName', 'RMSE');
ylabel('ToA RMSE (OFDM samples)');
ylim([0 max(0.25, 1.15*max(toa_rmse_delay))]);
yyaxis right;
plot(timing_path_delays, 100*toa_hit_delay, '--s', 'LineWidth', 1.5, ...
    'DisplayName', 'Correct detection');
ylabel('Correct detection (%)');
ylim([0 105]);
grid on;
xlabel('Second-path delay (OFDM samples)');
title(sprintf('Two-path channel sweep, RX SNR = %d dB', timing_fixed_snr));
xlim([timing_path_delays(1) timing_path_delays(end)]);

subplot(1,2,2);
hold on;
colors = lines(length(timing_selected_delays));
for d_idx = 1:length(timing_selected_delays)
    plot(timing_snr_range, toa_rmse_snr(:,d_idx), '-o', 'LineWidth', 1.5, ...
        'Color', colors(d_idx,:), ...
        'DisplayName', sprintf('delay = %d', timing_selected_delays(d_idx)));
end
grid on;
xlabel('RX SNR (dB)');
ylabel('ToA RMSE (OFDM samples)');
title(sprintf('Oversampled preamble correlation, N_{ovs} = %d', Novs));
xlim([timing_snr_range(1) timing_snr_range(end)]);
legend('Location','northeast');
sgtitle('OFDM packet time-of-arrival estimation');
export_if_valid(fig2, 'Fig2_Cross_Correlation.pdf');

rng_state = rng;
example_snr_db = 0;
example_lead_samples = 48;
example_tail_samples = 96;
example_delay_ovs = path_delay * Novs;
example_true_start = example_lead_samples * Novs + 1;
example_tx = [zeros(example_lead_samples*Novs, 1); preamble_ovs; ...
    zeros(example_tail_samples*Novs + example_delay_ovs, 1)];
example_h = zeros(example_delay_ovs + 1, 1);
example_h(1) = 1;
example_h(end) = sqrt(0.5);
example_rx = add_awgn_rx_snr(filter(example_h, 1, example_tx), example_snr_db);
example_metric = abs(conv(example_rx(:), flipud(conj(preamble_ovs(:))), 'valid')) / ...
    length(preamble_ovs);
[~, example_est_start] = max(example_metric);
rng(rng_state);

example_half_window = 20 * Novs;
example_idx = max(1, example_true_start-example_half_window): ...
    min(length(example_metric), example_true_start+example_half_window);
example_offsets = (example_idx - example_true_start) / Novs;
example_est_offset = (example_est_start - example_true_start) / Novs;

fig7 = figure('Name','Fig7 - Cross-correlation timing example', ...
    'Color', 'w', 'Units', 'pixels', 'Position', [120 120 720 520]);
plot(example_offsets, example_metric(example_idx), 'b-', 'LineWidth', 1.2, ...
    'DisplayName', 'xcorr');
hold on;
xline(0, 'r--', 'LineWidth', 1.2, 'DisplayName', 'True');
xline(example_est_offset, 'g:', 'LineWidth', 1.5, 'DisplayName', 'Est');
grid on;
xlabel('Sample offset relative to true packet start (OFDM samples)');
ylabel('|xcorr|');
title(sprintf('Cross-correlation for OFDM packet timing, RX SNR = %d dB', ...
    example_snr_db));
legend('Location','northeast');
export_if_valid(fig7, 'Fig7_Cross_Correlation_Example.pdf');

%% STEP 3 - Channel estimation: raw LS vs denoised TD projection
fprintf('\n=== Step 3: Channel estimation MSE (raw vs denoised) ===\n');

ce_snr_range = 0:5:40;
N_ce_trials = 500;
MSE_raw = zeros(size(ce_snr_range));
MSE_den = zeros(size(ce_snr_range));

clean_pr = filter(h, 1, preamble_ofdm);
for s_idx = 1:length(ce_snr_range)
    snr_db = ce_snr_range(s_idx);
    err_raw = zeros(N_ce_trials, 1);
    err_den = zeros(N_ce_trials, 1);

    for t = 1:N_ce_trials
        rx_pr = add_awgn_rx_snr(clean_pr, snr_db);
        Yp = ofdm_rx(rx_pr(1:N+CP), N, CP, active);

        H_raw = Yp ./ preamble_syms;
        h_ml = F_sub \ H_raw;
        H_den = F_sub * h_ml;

        err_raw(t) = mean(abs(H_raw - H_true_active).^2);
        err_den(t) = mean(abs(H_den - H_true_active).^2);
    end

    MSE_raw(s_idx) = mean(err_raw);
    MSE_den(s_idx) = mean(err_den);
    fprintf('  RX SNR=%2d dB | raw MSE=%.2f dB | denoised MSE=%.2f dB\n', ...
        snr_db, 10*log10(MSE_raw(s_idx)), 10*log10(MSE_den(s_idx)));
end

fprintf('  Average denoising gain: %.1f dB (theory about 10log10(Nsc/L)=%.1f dB)\n', ...
    mean(10*log10(MSE_raw ./ MSE_den)), 10*log10(Nsc/L));

fig3 = figure('Name','Fig3 - Channel estimation MSE');
plot(ce_snr_range, 10*log10(MSE_raw), 'r--s', 'LineWidth', 1.7, ...
    'DisplayName', 'Raw LS estimate');
hold on;
plot(ce_snr_range, 10*log10(MSE_den), 'b-o', 'LineWidth', 1.7, ...
    'DisplayName', 'Denoised TD estimate');
grid on;
xlabel('RX SNR (dB)');
ylabel('MSE (dB)');
title('OFDM preamble channel estimation: raw LS vs denoised');
legend('Location','southwest');
xlim([ce_snr_range(1) ce_snr_range(end)]);
export_if_valid(fig3, 'Fig3_Channel_Denoising.pdf');

channel_plot_snr_db = 20;
rx_pr_plot = add_awgn_rx_snr(clean_pr, channel_plot_snr_db);
Yp_plot = ofdm_rx(rx_pr_plot(1:N+CP), N, CP, active);
H_raw_plot = Yp_plot ./ preamble_syms;
h_ml_plot = F_sub \ H_raw_plot;
H_den_plot = F_sub * h_ml_plot;
h_true_plot = zeros(L, 1);
h_true_plot(1:length(h)) = h;

fig6 = figure('Name','Fig6 - Channel estimate example', ...
    'Color', 'w', 'Units', 'pixels', 'Position', [100 100 950 720]);
subplot(2,1,1);
plot(active, abs(H_true_active), 'k-', 'LineWidth', 1.5, ...
    'DisplayName', 'True channel');
hold on;
plot(active, abs(H_raw_plot), 'r:', 'LineWidth', 1.0, ...
    'DisplayName', 'Raw LS estimate');
plot(active, abs(H_den_plot), 'b--', 'LineWidth', 1.4, ...
    'DisplayName', 'Denoised TD estimate');
grid on;
xlabel('Subcarrier');
ylabel('|H[k]|');
title(sprintf('Channel estimation comparison - RX SNR = %d dB', ...
    channel_plot_snr_db));
legend('Location','best');
xlim([active(1) active(end)]);

subplot(2,1,2);
tap_idx = 0:L-1;
stem(tap_idx, abs(h_true_plot), 'k', 'LineWidth', 1.4, ...
    'DisplayName', 'True taps');
hold on;
stem(tap_idx, abs(h_ml_plot), 'b.', 'LineWidth', 1.2, ...
    'DisplayName', 'Time-domain ML estimate');
grid on;
xlabel('Tap index');
ylabel('|h[l]|');
title('Time-domain channel estimate used for denoising');
legend('Location','northeast');
xlim([0 L-1]);
export_if_valid(fig6, 'Fig6_Channel_Estimate_Example.pdf');

%% STEP 4 - Unified synchronization structure: BER comparison
fprintf('\n=== Step 4: Unified synchronization BER comparison ===\n');

SNR_range = 0:5:30;
BER_perfect = zeros(size(SNR_range));
BER_raw_est = zeros(size(SNR_range));
BER_den_est = zeros(size(SNR_range));
n_packets = 100;
constellation_snr_db = 25;
max_constellation_points = 8000;
rx_constellation = [];
tx_constellation = [];

for s_idx = 1:length(SNR_range)
    snr_db = SNR_range(s_idx);
    errs_p = 0;
    errs_r = 0;
    errs_d = 0;
    total_bits = 0;

    for pkt = 1:n_packets
        data_tx = randi([0 M-1], Nsc, n_sym);
        data_syms = qam_mod(data_tx(:));
        data_syms = reshape(data_syms, Nsc, n_sym);

        tx_packet = preamble_ofdm;
        for m_idx = 1:n_sym
            tx_packet = [tx_packet; ofdm_tx(data_syms(:,m_idx), N, CP, active)]; %#ok<AGROW>
        end

        rx_clean = filter(h, 1, tx_packet);
        rx_pkt = add_awgn_rx_snr(rx_clean, snr_db);

        t0 = estimate_packet_start(rx_pkt, preamble_ofdm);  % 0-based packet start
        if t0 + N + CP > length(rx_pkt)
            continue;
        end

        rx_pr_blk = rx_pkt(t0 + 1 : t0 + N + CP);
        Yp = ofdm_rx(rx_pr_blk, N, CP, active);
        H_raw = Yp ./ preamble_syms;

        h_ml = F_sub \ H_raw;
        H_den = F_sub * h_ml;

        data_rx_p = zeros(Nsc, n_sym);
        data_rx_r = zeros(Nsc, n_sym);
        data_rx_d = zeros(Nsc, n_sym);
        n_decoded = 0;

        data_start = t0 + length(preamble_ofdm);
        for m_idx = 1:n_sym
            idx_s = data_start + (m_idx-1)*(N+CP) + 1;
            idx_e = idx_s + N + CP - 1;
            if idx_e > length(rx_pkt)
                break;
            end

            Y = ofdm_rx(rx_pkt(idx_s:idx_e), N, CP, active);
            eq_perfect = Y ./ H_true_active;
            eq_raw = Y ./ H_raw;
            eq_den = Y ./ H_den;
            data_rx_p(:,m_idx) = qam_demod(eq_perfect);
            data_rx_r(:,m_idx) = qam_demod(eq_raw);
            data_rx_d(:,m_idx) = qam_demod(eq_den);
            if snr_db == constellation_snr_db && numel(rx_constellation) < max_constellation_points
                n_keep = min(numel(eq_den), max_constellation_points - numel(rx_constellation));
                rx_constellation = [rx_constellation; eq_den(1:n_keep)]; %#ok<AGROW>
                tx_constellation = [tx_constellation; data_syms(1:n_keep,m_idx)]; %#ok<AGROW>
            end
            n_decoded = n_decoded + 1;
        end

        if n_decoded == 0
            continue;
        end

        tx_ref = data_tx(:,1:n_decoded);
        rx_ref_p = data_rx_p(:,1:n_decoded);
        rx_ref_r = data_rx_r(:,1:n_decoded);
        rx_ref_d = data_rx_d(:,1:n_decoded);
        errs_p = errs_p + biterr(tx_ref(:), rx_ref_p(:), bits_per_sym);
        errs_r = errs_r + biterr(tx_ref(:), rx_ref_r(:), bits_per_sym);
        errs_d = errs_d + biterr(tx_ref(:), rx_ref_d(:), bits_per_sym);
        total_bits = total_bits + numel(tx_ref) * bits_per_sym;
    end

    BER_perfect(s_idx) = errs_p / total_bits;
    BER_raw_est(s_idx) = errs_r / total_bits;
    BER_den_est(s_idx) = errs_d / total_bits;

    fprintf('  RX SNR=%2d dB | perfect=%.4g | raw=%.4g | denoised=%.4g\n', ...
        snr_db, BER_perfect(s_idx), BER_raw_est(s_idx), BER_den_est(s_idx));
end

fig4 = figure('Name','Fig4 - Unified BER');
semilogy(SNR_range, max(BER_perfect, plot_floor), 'k-o', 'LineWidth', 1.6, ...
    'DisplayName', 'Perfect CSI reference');
hold on;
semilogy(SNR_range, max(BER_raw_est, plot_floor), 'r--s', 'LineWidth', 1.6, ...
    'DisplayName', 'Raw LS channel estimate');
semilogy(SNR_range, max(BER_den_est, plot_floor), 'b-.^', 'LineWidth', 1.6, ...
    'DisplayName', 'Denoised TD estimate');
grid on;
xlabel('RX SNR (dB)');
ylabel('Bit Error Rate (BER)');
title(sprintf('Unified OFDM synchronization - %d-QAM, two-path channel', M));
legend('Location','southwest');
ylim([plot_floor 1]);
xlim([SNR_range(1) SNR_range(end)]);
export_if_valid(fig4, 'Fig4_Unified_BER.pdf');

fig5 = figure('Name','Fig5 - Synchronized received constellation', ...
    'Color', 'w', 'Units', 'pixels', 'Position', [120 120 700 650]);
plot(real(rx_constellation), imag(rx_constellation), '.', ...
    'Color', [0 0.45 0.74], 'MarkerSize', 6, ...
    'DisplayName', 'Received after sync + denoised equalization');
hold on;
ideal_constellation = qam_mod((0:M-1).');
plot(real(ideal_constellation), imag(ideal_constellation), 'kx', ...
    'LineWidth', 1.6, 'MarkerSize', 10, 'DisplayName', 'Ideal 16-QAM points');
grid on;
axis equal;
xlabel('In-phase');
ylabel('Quadrature');
title(sprintf('Synchronized received QAM constellation - RX SNR = %d dB', ...
    constellation_snr_db));
legend('Location','best');
export_if_valid(fig5, 'Fig5_Synchronized_Constellation.pdf');

fprintf('\nDone. Seven report figures generated.\n');

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

function y = add_awgn_rx_snr(x, snr_db)
    snr_lin = 10^(snr_db/10);
    noise_var = mean(abs(x).^2) / snr_lin;
    y = x + sqrt(noise_var/2) * (randn(size(x)) + 1j*randn(size(x)));
end

function x_ovs = oversample_signal(x, Novs)
    x_ovs = interpft(x(:), length(x) * Novs) * sqrt(Novs);
end

function [rmse_samples, bias_samples, hit_rate] = ofdm_toa_stats( ...
    preamble_ovs, Novs, path_delay, snr_db, n_trials, lead_samples, tail_samples)
    lead_ovs = lead_samples * Novs;
    tail_ovs = tail_samples * Novs;
    delay_ovs = path_delay * Novs;
    true_start = lead_ovs + 1;
    tx_ovs = [zeros(lead_ovs, 1); preamble_ovs(:); ...
        zeros(tail_ovs + delay_ovs, 1)];
    err_samples = zeros(n_trials, 1);

    for t = 1:n_trials
        second_path_phase = exp(1j * 2*pi*rand);
        h_ovs = zeros(delay_ovs + 1, 1);
        h_ovs(1) = 1;
        h_ovs(end) = sqrt(0.5) * second_path_phase;

        rx_clean = filter(h_ovs, 1, tx_ovs);
        rx = add_awgn_rx_snr(rx_clean, snr_db);
        metric = abs(conv(rx(:), flipud(conj(preamble_ovs(:))), 'valid')) / ...
            length(preamble_ovs);
        [~, n_hat] = max(metric);
        err_samples(t) = (n_hat - true_start) / Novs;
    end

    rmse_samples = sqrt(mean(err_samples.^2));
    bias_samples = mean(err_samples);
    hit_rate = mean(abs(err_samples) <= 1/Novs);
end

function [start_hat, metric] = estimate_packet_start(rx, preamble)
    metric = abs(conv(rx(:), flipud(conj(preamble(:))), 'valid'));
    [~, idx] = max(metric);
    start_hat = idx - 1;  % 0-based offset, easier to compare with inserted zeros
end

function export_if_valid(fig, filename)
    drawnow;
    if ~isgraphics(fig, 'figure')
        warning('Skipping export of %s because the figure was closed or deleted.', filename);
        return;
    end

    try
        exportgraphics(fig, filename, 'ContentType', 'vector');
    catch ME
        warning('exportgraphics failed for %s (%s). Trying print instead.', ...
            filename, ME.message);
        if isgraphics(fig, 'figure')
            print(fig, filename, '-dpdf', '-vector');
        end
    end
end
