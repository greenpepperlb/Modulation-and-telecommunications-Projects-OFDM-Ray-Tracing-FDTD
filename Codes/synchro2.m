%% =========================================================================
%  ELEC-H401 - Synchronization
%  Etape 1 : Impact du phase offset et du time shift sur le BER
%
%  Contexte : On part de la chaine RRC parfaite (partie 4).
%  On introduit ARTIFICIELLEMENT des erreurs de synchronisation pour
%  mesurer leur impact AVANT de les corriger avec Gardner + Xcorr.
%
%  Structure du script :
%   Section A : Parametres communs
%   Section B : Chaine de reference (parfaite, sans erreur)
%   Section C : Impact du phase offset
%   Section D : Impact du time shift
% =========================================================================

clear; close all; clc;

%% =========================================================================
%  SECTION A : PARAMETRES COMMUNS
% =========================================================================

% --- Parametres de la constellation ---
QAM_order       = 16;           % On monte a 16-QAM pour mieux voir la degradation
bits_per_symbol = log2(QAM_order);
nb_symbols      = 50000;        % Nombre de symboles
N_bits          = nb_symbols * bits_per_symbol;

% --- Parametres physiques ---
roll_off        = 0.3;
symbol_rate     = 5e6;          % Rs = 5 Msymboles/sec
sample_rate     = 60e6;         % Fs = 60 Msamples/sec
M               = sample_rate / symbol_rate;  % M = 12 samples/symbole
RRC_length      = 20;           % Longueur du filtre en nombre de symboles

% --- Plage de Eb/N0 ---
EbN0_dB_range = 0:2:20;

% --- Valeurs d'offsets a tester ---
% Phase offsets (en radians)
phase_offsets = [0, pi/16, pi/8, pi/4];
phase_labels  = {'0', '\pi/16', '\pi/8', '\pi/4'};

% Time shifts (en fraction de T)
% On plafonne a 25% de T : au dela c'est une bouillie de symboles,
% aucun systeme reel ne fonctionnerait dans ces conditions.
% linspace(0, 0.25, 10) puis deduplication en samples entiers.
time_shifts_raw = linspace(0, 0.25, 10);
deltas_raw      = round(time_shifts_raw * M);
deltas_unique   = unique(deltas_raw);          % supprime les doublons
time_shifts     = deltas_unique / M;           % repasse en fraction de T
time_labels     = arrayfun(@(x) sprintf('%.0f%%', x*100), time_shifts, ...
                           'UniformOutput', false);

%% =========================================================================
%  SECTION B : GENERATION DU SIGNAL EMIS (commune a toutes les simulations)
% =========================================================================
% On genere UNE SEULE FOIS le signal emis car il ne change pas.
% Ce qui change c'est ce qu'on fait avec le signal RECU.

% 1. Generation des bits et mapping QAM
random_bits       = randi([0 1], N_bits, 1);
modulated_symbols = qammod(random_bits, QAM_order, ...
                           'InputType', 'bit', 'UnitAveragePower', true);

% 2. Filtre RRC (genere une seule fois)
h_rrc = generate_rrc_filter(sample_rate, symbol_rate, roll_off, RRC_length);

% Normalisation de l'energie du filtre
% Important : sans ca, la puissance du signal change apres filtrage
h_rrc = h_rrc / norm(h_rrc);

% 3. Upsampling + filtrage TX
symbols_upsampled = upsample(modulated_symbols, M);
tx_signal         = conv(symbols_upsampled, h_rrc, 'same');

% Puissance du signal emis (mesuree une seule fois, elle ne change pas)
signal_power = mean(abs(tx_signal).^2);

% BER theorique (reference)
ber_theoretical = berawgn(EbN0_dB_range, 'qam', QAM_order);

%% =========================================================================
%  SECTION C : IMPACT DU PHASE OFFSET
%
%  On garde le downsampling PARFAIT (pas de time shift).
%  On introduit uniquement une rotation e^(j*phi) sur le signal recu
%  apres le matched filter.
%
%  Physiquement : le phase offset tourne toute la constellation d'un angle
%  phi. Les frontieres de decision (verticales/horizontales en QAM) ne
%  correspondent plus aux vrais symboles -> erreurs.
% =========================================================================

fprintf('=== Section C : Impact du phase offset ===\n');

% Matrice pour stocker les BER : lignes = EbN0, colonnes = phase offsets
ber_phase = zeros(length(EbN0_dB_range), length(phase_offsets));

snr_dB_all = EbN0_dB_range + 10*log10(bits_per_symbol) - 10*log10(M);
snr_lin_all = 10.^(snr_dB_all / 10);
noise_sigma_all = sqrt(signal_power ./ snr_lin_all / 2);
phase_rot = exp(1j * phase_offsets);

for p = 1:length(phase_offsets)
    phi = phase_offsets(p);
    fprintf('  Phase offset = %s rad\n', phase_labels{p});

    for i = 1:length(EbN0_dB_range)
        % --- Canal AWGN ---
        noise_sigma = noise_sigma_all(i);  % /2 car bruit complexe (I et Q)
        noise       = noise_sigma * (randn(size(tx_signal)) + ...
                                   1i * randn(size(tx_signal)));
        rx_signal   = tx_signal + noise;

        % --- Matched filter (filtre RRC recepteur) ---
        rx_filtered = conv(rx_signal, h_rrc, 'same');

        % --- Downsampling PARFAIT (pas de time shift ici) ---
        % On prend le sample au centre de chaque symbole.
        % Le delai du filtre RRC est RRC_length/2 * M samples.
        % Pour simplifier on prend l'index 1 (comme dans ta chaine originale).
        received_symbols = rx_filtered(1:M:end);
        received_symbols = received_symbols(1:nb_symbols);

        % --- APPLICATION DU PHASE OFFSET ---
        % C'est ICI qu'on introduit l'erreur de synchronisation.
        % Dans la realite, ce phase offset est involontaire (difference
        % d'oscillateurs). Ici on le simule volontairement pour mesurer
        % son impact.
        received_symbols_offset = received_symbols * phase_rot(p);

        % --- Demodulation et BER ---
        demod_bits = qamdemod(received_symbols_offset, QAM_order, ...
                              'OutputType', 'bit', 'UnitAveragePower', true);
        num_errors         = sum(random_bits ~= demod_bits);
        ber_phase(i, p)    = num_errors / N_bits;
    end
end

% --- Figure : BER vs Eb/N0 pour differents phase offsets ---
figure(1); clf;
semilogy(EbN0_dB_range, ber_theoretical, 'k--', 'LineWidth', 2, ...
         'DisplayName', 'Theorique (ideal)');
hold on;
colors = {'b', 'r', 'm', 'g'};
for p = 1:length(phase_offsets)
    semilogy(EbN0_dB_range, ber_phase(:,p), ...
             [colors{p} 'o-'], 'LineWidth', 1.5, ...
             'DisplayName', ['\phi = ' phase_labels{p} ' rad']);
end
grid on;
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate (BER)');
title('Impact du phase offset sur le BER (QAM-16)');
legend('Location', 'southwest');
ylim([1e-4 1]);

%% =========================================================================
%  SECTION D : IMPACT DU TIME SHIFT
%
%  On garde le phase offset = 0.
%  On decale l'instant de downsampling de delta samples.
%
%  Physiquement : au lieu d'echantillonner au sommet de la cloche RC
%  (zero ISI), on echantillonne sur le flanc -> les queues des symboles
%  voisins ne valent plus zero -> ISI -> erreurs.
%
%  Implementation : au lieu de rx_filtered(1:M:end)
%                   on prend    rx_filtered(1+delta:M:end)
%  ou delta = round(epsilon * M) est l'erreur en samples entiers.
% =========================================================================

fprintf('\n=== Section D : Impact du time shift ===\n');

% Les deltas sont deja calcules proprement dans les parametres
fprintf('  Time shifts en samples : %s\n', num2str(deltas_unique));

% Matrice pour stocker les BER
ber_time = zeros(length(EbN0_dB_range), length(time_shifts));

for t = 1:length(time_shifts)
    delta = deltas_unique(t);
    fprintf('  Time shift = %s de T = %d samples\n', ...
            time_labels{t}, delta);

    for i = 1:length(EbN0_dB_range)
        % --- Canal AWGN (identique a la section C) ---
        noise_sigma = noise_sigma_all(i);
        noise       = noise_sigma * (randn(size(tx_signal)) + ...
                                   1i * randn(size(tx_signal)));
        rx_signal   = tx_signal + noise;

        % --- Matched filter ---
        rx_filtered = conv(rx_signal, h_rrc, 'same');

        % --- Downsampling DECALE de delta samples ---
        % C'est ICI qu'on introduit le time shift.
        % Au lieu du sommet parfait (index 1), on prend (1 + delta).
        % Si delta = 0 -> parfait (reference)
        % Si delta > 0 -> on echantillonne trop tard (a droite du sommet)
        start_idx        = 1 + delta;
        received_symbols = rx_filtered(start_idx:M:end);
        received_symbols = received_symbols(1:nb_symbols);

        % --- Demodulation et BER (pas de phase offset ici) ---
        demod_bits      = qamdemod(received_symbols, QAM_order, ...
                                   'OutputType', 'bit', 'UnitAveragePower', true);
        num_errors      = sum(random_bits ~= demod_bits);
        ber_time(i, t)  = num_errors / N_bits;
    end
end

% --- Figure unique : toutes les courbes sur un seul graphe ---
cmap = lines(length(time_shifts));

figure(2); clf;
semilogy(EbN0_dB_range, ber_theoretical, 'k--', 'LineWidth', 2, ...
         'DisplayName', 'Theorique (ideal)');
hold on;
for t = 1:length(time_shifts)
    semilogy(EbN0_dB_range, ber_time(:,t), 's-', ...
             'Color', cmap(t,:), 'LineWidth', 1.5, ...
             'DisplayName', ['\epsilon = ' time_labels{t}]);
end
grid on;
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate (BER)');
title('Impact du time shift sur le BER (QAM-16)');
legend('Location', 'southwest', 'FontSize', 8);
ylim([1e-4 1]);

%% =========================================================================
%  SECTION E : VISUALISATION DES CONSTELLATIONS (a haut Eb/N0 = 20 dB)
%
%  On visualise l'effet des offsets directement sur la constellation.
%  C'est le moyen le plus intuitif de comprendre ce qui se passe.
% =========================================================================

EbN0_visu = 20;  % dB - haut SNR pour voir clairement l'effet geometrique

snr_dB_visu  = EbN0_visu + 10*log10(bits_per_symbol) - 10*log10(M);
snr_lin_visu = 10^(snr_dB_visu / 10);
noise_power_visu = signal_power / snr_lin_visu;
noise_sigma_visu = sqrt(noise_power_visu / 2);
noise_visu       = noise_sigma_visu * (randn(size(tx_signal)) + ...
                                      1i * randn(size(tx_signal)));
rx_visu          = tx_signal + noise_visu;
rx_filt_visu     = conv(rx_visu, h_rrc, 'same');

% Symboles parfaitement echantillonnes (reference)
syms_perfect = rx_filt_visu(1:M:end);
syms_perfect = syms_perfect(1:nb_symbols);

% Symboles avec phase offset de pi/8
syms_phase   = syms_perfect * exp(1j * pi/8);

% Symboles avec time shift de ~33% de T (le plus grand disponible)
delta_visu   = deltas_unique(2);
syms_time    = rx_filt_visu(1+delta_visu:M:end);
syms_time    = syms_time(1:nb_symbols);

N_plot = 5000;  % Nombre de symboles a afficher

figure(3); clf;
subplot(1,3,1);
plot(real(syms_perfect(1:N_plot)), imag(syms_perfect(1:N_plot)), 'b.', 'MarkerSize', 4);
title('Reference (parfait)');
axis([-2 2 -2 2]); grid on; axis square;
xlabel('I'); ylabel('Q');

subplot(1,3,2);
plot(real(syms_phase(1:N_plot)), imag(syms_phase(1:N_plot)), 'r.', 'MarkerSize', 4);
title('Phase offset = \pi/8');
axis([-2 2 -2 2]); grid on; axis square;
xlabel('I'); ylabel('Q');

subplot(1,3,3);
plot(real(syms_time(1:N_plot)), imag(syms_time(1:N_plot)), 'm.', 'MarkerSize', 4);
title(['Time shift = ' sprintf('%.0f', deltas_unique(end)/M*100) '% de T']);
axis([-2 2 -2 2]); grid on; axis square;
xlabel('I'); ylabel('Q');

sgtitle(['Constellations QAM-16 a Eb/N0 = ' num2str(EbN0_visu) ' dB']);

fprintf('\n=== Simulation terminee ===\n');
fprintf('Figure 1 : BER vs Eb/N0 pour differents phase offsets\n');
fprintf('Figure 2 : BER vs Eb/N0 pour differents time shifts\n');
fprintf('Figure 3 : Visualisation des constellations\n');

%% =========================================================================
%  STEP 2 : Algorithme de Gardner (BOUCLE FERMEE - slide 42)
%  Ajoute en fin de fichier dans une fonction locale pour ne pas ecraser
%  les parametres de l'etape 1.
% =========================================================================
fprintf('\n=== Step 2 : Gardner (boucle fermee) ===\n');
run_gardner_step2(QAM_order, bits_per_symbol, nb_symbols, N_bits, M, ...
                  roll_off, random_bits, tx_signal, signal_power, h_rrc,RRC_length);

%% =========================================================================
%  STEP 3 : Cross-correlation pour ToA et phase offset
% =========================================================================
fprintf('\n=== Step 3 : Cross-correlation ToA et phase ===\n');
run_xcorr_step3(QAM_order, bits_per_symbol);

%% -- LOCAL FUNCTIONS --
function run_gardner_step2(QAM_order, bits_per_symbol, nb_symbols, N_bits, M, ...
                           roll_off, random_bits, tx_signal, signal_power, h_rrc,RRC_length)

    mu_values    = [0.005, 0.02, 0.05];
    kappa_labels = {'\kappa = 0.005', '\kappa = 0.02', '\kappa = 0.05'};

    epsilon_true = 0.20;
    EbN0_dB_RMSE = -5:2:15;
    EbN0_dB_BER  = 0:2:20;
    N_warmup     = 1000;

    snr_dB_rmse = EbN0_dB_RMSE + 10*log10(bits_per_symbol) - 10*log10(M);
    snr_lin_rmse = 10.^(snr_dB_rmse / 10);
    noise_sigma_rmse = sqrt(signal_power ./ snr_lin_rmse / 2);

    snr_dB_ber = EbN0_dB_BER + 10*log10(bits_per_symbol) - 10*log10(M);
    snr_lin_ber = 10.^(snr_dB_ber / 10);
    noise_sigma_ber = sqrt(signal_power ./ snr_lin_ber / 2);

    rx_filt_clean = conv(tx_signal, h_rrc, 'same');

    % Convergence du Gardner (sans bruit)
    fprintf('=== Section D : Convergence du Gardner (sans bruit) ===\n');

    eps_hist = zeros(nb_symbols-1, length(mu_values));
    for m_idx = 1:length(mu_values)
        mu = mu_values(m_idx);
        [eps_hat, ~] = gardner_closed_loop(rx_filt_clean, M, mu, epsilon_true, nb_symbols, RRC_length);
        eps_hist(1:length(eps_hat), m_idx) = eps_hat;
        fprintf('  %s : epsilon_hat final = %.4f (vrai = %.4f)\n', ...
            kappa_labels{m_idx}, mean(eps_hat(end-200:end)), epsilon_true);
    end

    cmap_mu = [0    0.45 0.74; 0.85 0.33 0.10; 0.5  0.5  0.5];
    lw_values = [1.8, 1.5, 0.4];
    res_hist = epsilon_true - eps_hist;

    figure('Name', 'Step 2 - Gardner Convergence'); hold on;
    n_plot = min(20000, nb_symbols - 1); % longer horizon for small kappa
    for m_idx = length(mu_values):-1:1
        plot(1:n_plot, res_hist(1:n_plot, m_idx), ...
             'LineWidth', lw_values(m_idx), ...
             'Color', cmap_mu(m_idx,:), ...
             'DisplayName', kappa_labels{m_idx});
    end
    yline(0, 'k--', 'LineWidth', 1.5, 'DisplayName', 'residu = 0');
    grid on;
    xlabel('Indice symbole n');
    ylabel('erreur residuelle (epsilon_vrai - epsilon_hat)');
    title(sprintf('Erreur residuelle du Gardner — QAM-%d, epsilon=%.2f, sans bruit', ...
                  QAM_order, epsilon_true));
    legend('Location', 'southeast');
    ylim([-0.1 0.25]);

    % Convergence moyenne (Monte Carlo) pour lisser la courbe
    fprintf('\n=== Section D2 : Convergence moyenne (Monte Carlo) ===\n');
    n_avg = 500;  % augmente a 1000 pour un lissage plus fort (plus lent)
    n_plot = min(20000, nb_symbols - 1); % augmenter si besoin de convergence
    nb_symbols_avg = n_plot + 1;
    N_bits_avg = nb_symbols_avg * bits_per_symbol;

    res_avg = zeros(n_plot, length(mu_values));
    res_std = zeros(n_plot, length(mu_values));
    for m_idx = 1:length(mu_values)
        mu = mu_values(m_idx);
        sum_eps = zeros(n_plot, 1);
        sum_eps2 = zeros(n_plot, 1);
        for r = 1:n_avg
            bits_mc = randi([0 1], N_bits_avg, 1);
            syms_mc = qammod(bits_mc, QAM_order, ...
                             'InputType', 'bit', 'UnitAveragePower', true);
            tx_mc = conv(upsample(syms_mc, M), h_rrc, 'same');
            rx_mc = conv(tx_mc, h_rrc, 'same');

            [eps_hat, ~] = gardner_closed_loop(rx_mc, M, mu, epsilon_true, nb_symbols_avg, RRC_length);
            n_eff = min(n_plot, length(eps_hat));
            sum_eps(1:n_eff) = sum_eps(1:n_eff) + eps_hat(1:n_eff);
            sum_eps2(1:n_eff) = sum_eps2(1:n_eff) + eps_hat(1:n_eff).^2;
        end
        mean_eps = sum_eps / n_avg;
        var_eps = max(sum_eps2 / n_avg - mean_eps.^2, 0);
        res_avg(:,m_idx) = epsilon_true - mean_eps;
        res_std(:,m_idx) = sqrt(var_eps);
    end

        kappa_labels = arrayfun(@(v) ['\kappa = ' num2str(v)], mu_values, ...
                   'UniformOutput', false);
        avg_idx = [1 2]; % exclude kappa = 0.05 from the average plot

        figure('Name', 'Step 2 - Gardner Convergence (Average)'); hold on;
        for m_idx = fliplr(avg_idx)
           plot(1:n_plot, res_avg(:,m_idx), ...
               'LineWidth', lw_values(m_idx), ...
               'Color', cmap_mu(m_idx,:), ...
               'DisplayName', kappa_labels{m_idx});
           plot(1:n_plot, res_avg(:,m_idx) + 3*res_std(:,m_idx), '--', ...
               'Color', cmap_mu(m_idx,:), 'HandleVisibility', 'off');
           plot(1:n_plot, res_avg(:,m_idx) - 3*res_std(:,m_idx), '--', ...
               'Color', cmap_mu(m_idx,:), 'HandleVisibility', 'off');
        end
    yline(0, 'k--', 'LineWidth', 1.5, 'DisplayName', 'residu = 0');
    grid on;
    xlabel('Indice symbole n');
    ylabel('erreur residuelle moyenne (\pm 3\sigma)');
    title(sprintf('Erreur residuelle moyenne du Gardner (\\pm 3\\sigma) — QAM-%d, epsilon=%.2f, %d realisations', ...
                  QAM_order, epsilon_true, n_avg));
    legend('Location', 'southeast');
    ylim([-0.1 0.25]);

    % RMSE apres convergence vs SNR
    fprintf('\n=== Section E : RMSE apres convergence vs SNR ===\n');

    rmse_mat = zeros(length(EbN0_dB_RMSE), length(mu_values));
    for m_idx = 1:length(mu_values)
        mu = mu_values(m_idx);
        fprintf('  %s ...\n', kappa_labels{m_idx});
        for i = 1:length(EbN0_dB_RMSE)
            noise_sigma = noise_sigma_rmse(i);
            noise       = noise_sigma * (randn(size(tx_signal)) + ...
                                        1i * randn(size(tx_signal)));
            rx_filt     = conv(tx_signal + noise, h_rrc, 'same');
            [eps_hat, ~] = gardner_closed_loop(rx_filt, M, mu, epsilon_true, nb_symbols, RRC_length);
            eps_post    = eps_hat(N_warmup:end);
            rmse_mat(i, m_idx) = sqrt(mean((eps_post - epsilon_true).^2));
        end
    end

    figure('Name', 'Step 2 - RMSE vs SNR'); hold on;
    for m_idx = 1:length(mu_values)
        semilogy(EbN0_dB_RMSE, rmse_mat(:,m_idx), 'o-', ...
                 'Color', cmap_mu(m_idx,:), 'LineWidth', 1.5, ...
                 'DisplayName', kappa_labels{m_idx});
    end
    set(gca, 'YScale', 'log');
    grid on;
    xlabel('RX SNR (dB)');
    ylabel('RMSE normalise de epsilon');
    title(sprintf('RMSE du Gardner apres convergence vs SNR (QAM-%d, roll-off=%.2f)', ...
                  QAM_order, roll_off));
    legend('Location', 'northeast');
    ylim([1e-3 1e3]);

    % BER avec et sans correction Gardner
    fprintf('\n=== Section F : BER avec et sans Gardner ===\n');

    mu_best     = 0.02;
    ber_no_sync = zeros(length(EbN0_dB_BER), 1);
    ber_gardner = zeros(length(EbN0_dB_BER), 1);
    ber_perfect = zeros(length(EbN0_dB_BER), 1);
    ber_theory  = berawgn(EbN0_dB_BER, 'qam', QAM_order);

    for i = 1:length(EbN0_dB_BER)
        fprintf('  Eb/N0 = %d dB\n', EbN0_dB_BER(i));

        noise_sigma = noise_sigma_ber(i);
        noise       = noise_sigma * (randn(size(tx_signal)) + ...
                                    1i * randn(size(tx_signal)));
        rx_filt     = conv(tx_signal + noise, h_rrc, 'same');

        syms_perf = rx_filt(1:M:end);
        n_perf    = min(length(syms_perf), nb_symbols);
        bits_perf = qamdemod(syms_perf(1:n_perf), QAM_order, ...
                             'OutputType','bit','UnitAveragePower',true);
        ber_perfect(i) = sum(random_bits(1:n_perf*bits_per_symbol) ~= bits_perf) / N_bits;

        delta_int = round(epsilon_true * M);
        syms_ns   = rx_filt(1+delta_int:M:end);
        n_ns      = min(length(syms_ns), nb_symbols);
        bits_ns   = qamdemod(syms_ns(1:n_ns), QAM_order, ...
                             'OutputType','bit','UnitAveragePower',true);
        ber_no_sync(i) = sum(random_bits(1:n_ns*bits_per_symbol) ~= bits_ns) / N_bits;

        [~, syms_gard] = gardner_closed_loop(rx_filt, M, mu_best, epsilon_true, nb_symbols, RRC_length);
        n_gard    = min(length(syms_gard), nb_symbols);
        bits_gard = qamdemod(syms_gard(1:n_gard), QAM_order, ...
                             'OutputType','bit','UnitAveragePower',true);
        ber_gardner(i) = sum(random_bits(1:n_gard*bits_per_symbol) ~= bits_gard) / N_bits;
    end

    figure('Name', 'Step 2 - BER with Gardner');
    semilogy(EbN0_dB_BER, ber_theory,  'k--', 'LineWidth', 2,   'DisplayName', 'Theorique');
    hold on;
    semilogy(EbN0_dB_BER, ber_perfect, 'g^-', 'LineWidth', 1.5, 'DisplayName', 'Parfait (ref)');
    semilogy(EbN0_dB_BER, ber_no_sync, 'rx-', 'LineWidth', 1.5, ...
             'DisplayName', sprintf('Sans sync (eps=%.0f%% T)', epsilon_true*100));
    semilogy(EbN0_dB_BER, ber_gardner, 'bo-', 'LineWidth', 1.5, ...
             'DisplayName', sprintf('Avec Gardner (\\kappa=%.2f)', mu_best));
    grid on;
    xlabel('Eb/N0 (dB)');
    ylabel('Bit Error Rate (BER)');
    title(sprintf('BER avec et sans correction Gardner (QAM-%d)', QAM_order));
    legend('Location', 'southwest');
    ylim([1e-4 1]);

    fprintf('\n=== Simulation terminee (Step 2) ===\n');
end

function run_xcorr_step3(QAM_order, bits_per_symbol)
    % Parametres
    N_preamble_values = [4, 8, 16];
    N_labels          = {'N = 4', 'N = 8', 'N = 16'};

    N_data        = 200;
    N_trames      = 200;
    EbN0_dB_range = -5:2:15;
    phi_true      = pi/4;
    n_true        = 50;

    fprintf('=== Configuration ===\n');
    fprintf('  QAM order             : %d\n', QAM_order);
    fprintf('  Longueurs de preamble : %s\n', mat2str(N_preamble_values));
    fprintf('  Symboles par trame    : %d\n', N_data);
    fprintf('  Position vraie ToA    : %d\n', n_true);
    fprintf('  Phase vraie           : %.3f rad (%.1f deg)\n', phi_true, phi_true*180/pi);
    fprintf('  Realisations par pt   : %d\n', N_trames);
    fprintf('  Plage Eb/N0           : %d a %d dB\n', EbN0_dB_range(1), EbN0_dB_range(end));

    % Generation du preamble
    rng(123);
    N_max         = max(N_preamble_values);
    preamble_bits = randi([0 1], N_max * 2, 1);
    preamble_full = qammod(preamble_bits, 4, ...
                           'InputType','bit','UnitAveragePower',true);

    % Exemple de cross-correlation (illustration)
    fprintf('\n=== Section C : Exemple de cross-correlation ===\n');

    N_demo    = 16;
    preamble  = preamble_full(1:N_demo);
    EbN0_demo = 10;

    fprintf('  Parametres demo : N = %d, Eb/N0 = %d dB\n', N_demo, EbN0_demo);

    N_before     = n_true - 1;
    N_after      = N_data - N_demo - N_before;
    data_before  = qammod(randi([0 1], N_before * bits_per_symbol, 1), ...
                          QAM_order, 'InputType','bit','UnitAveragePower',true);
    data_after   = qammod(randi([0 1], N_after * bits_per_symbol, 1), ...
                          QAM_order, 'InputType','bit','UnitAveragePower',true);
    trame_clean  = [data_before; preamble; data_after];

    fprintf('  Trame construite : %d symboles total\n', length(trame_clean));
    fprintf('     - %d symboles avant preamble\n', N_before);
    fprintf('     - %d symboles preamble (positions %d a %d)\n', ...
            N_demo, n_true, n_true + N_demo - 1);
    fprintf('     - %d symboles apres preamble\n', N_after);

    trame_rotated = trame_clean * exp(1j * phi_true);
    snr_sym_dB    = EbN0_demo + 10*log10(bits_per_symbol);
    snr_lin       = 10^(snr_sym_dB / 10);
    sig_power     = mean(abs(trame_rotated).^2);
    noise_sigma   = sqrt(sig_power / snr_lin / 2);
    trame_recue   = trame_rotated + noise_sigma * ...
                    (randn(size(trame_rotated)) + 1i*randn(size(trame_rotated)));

    fprintf('  Puissance signal = %.3f, sigma bruit = %.3f\n', sig_power, noise_sigma);

    C = conv(trame_recue, flipud(conj(preamble)), 'valid') / N_demo;
    N_search = length(C);

    [max_module, n_hat] = max(abs(C));
    phi_hat             = -angle(C(n_hat));

    expected_peak  = 1;
    expected_floor = 1 / sqrt(N_demo);

    fprintf('  Resultats :\n');
    fprintf('     ToA vrai    = %d, ToA estime = %d (erreur = %d sample)\n', ...
            n_true, n_hat, n_hat - n_true);
    fprintf('     phi vrai    = %.3f rad, phi estime = %.3f rad (erreur = %.3f rad)\n', ...
            phi_true, phi_hat, wrap_to_pi(phi_hat - phi_true));
    fprintf('     |C| max     = %.3f (attendu sans bruit : %.3f)\n', max_module, expected_peak);
    fprintf('     |C| moyen   = %.3f (attendu hors pic   : %.3f)\n', ...
            mean(abs(C([1:n_hat-5, n_hat+5:end]))), expected_floor);
    fprintf('     SNR de pic  = %.1f (peak/floor)\n', ...
            max_module / mean(abs(C([1:n_hat-5, n_hat+5:end]))));

    % Use a higher base index to avoid overwriting Step 2 figures.
    fig_base = 10;
    figure(fig_base + 1); clf;
    subplot(2,1,1);
    plot(1:N_search, abs(C), 'b-', 'LineWidth', 1.2); hold on;
    xline(n_true, 'g--', 'LineWidth', 2);
    xline(n_hat,  'r:',  'LineWidth', 2);
    grid on;
    xlabel('Position n');
    ylabel('|C[n]|');
    title('Module de la cross-correlation');
    legend('|C[n]|', 'ToA vrai', 'ToA estime');

    subplot(2,1,2);
    plot(1:N_search, angle(C), 'b.', 'MarkerSize', 4); hold on;
    plot(n_hat, angle(C(n_hat)), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    yline(-phi_true, 'g--', 'LineWidth', 2);
    grid on;
    xlabel('Position n');
    ylabel('angle(C[n]) (rad)');
    title('Phase de la cross-correlation (en n_hat c''est -phi)');
    legend('angle(C[n])', 'En n_hat', '-phi vrai');

    sgtitle(sprintf('Exemple de cross-correlation, N=%d, Eb/N0=%d dB', N_demo, EbN0_demo));

    % RMSE du ToA et de la phase en fonction du SNR
    fprintf('\n=== Section D : RMSE vs SNR pour differentes N ===\n');

    rmse_toa = zeros(length(EbN0_dB_range), length(N_preamble_values));
    rmse_phi = zeros(length(EbN0_dB_range), length(N_preamble_values));
    detection_rate = zeros(length(EbN0_dB_range), length(N_preamble_values));

    snr_sym_dB_all = EbN0_dB_range + 10*log10(bits_per_symbol);
    snr_lin_all = 10.^(snr_sym_dB_all / 10);

    for k = 1:length(N_preamble_values)
        N_demo   = N_preamble_values(k);
        preamble = preamble_full(1:N_demo);
        fprintf('\n  N = %d :\n', N_demo);
        fprintf('    Eb/N0 | RMSE ToA | RMSE phi (rad) | Taux detection\n');
        fprintf('    ------|----------|----------------|---------------\n');

        for i = 1:length(EbN0_dB_range)
            snr_lin    = snr_lin_all(i);

            err_toa   = zeros(N_trames, 1);
            err_phi   = zeros(N_trames, 1);
            n_correct = 0;

            for t = 1:N_trames
                N_before    = n_true - 1;
                N_after     = N_data - N_demo - N_before;
                data_before = qammod(randi([0 1], N_before * bits_per_symbol, 1), ...
                                     QAM_order, 'InputType','bit', ...
                                     'UnitAveragePower',true);
                data_after  = qammod(randi([0 1], N_after * bits_per_symbol, 1), ...
                                     QAM_order, 'InputType','bit', ...
                                     'UnitAveragePower',true);
                trame_clean = [data_before; preamble; data_after];
                trame_rot   = trame_clean * exp(1j * phi_true);

                sig_power   = mean(abs(trame_rot).^2);
                noise_sigma = sqrt(sig_power / snr_lin / 2);
                trame_recue = trame_rot + noise_sigma * ...
                              (randn(size(trame_rot)) + 1i*randn(size(trame_rot)));

                C = conv(trame_recue, flipud(conj(preamble)), 'valid') / N_demo;

                [~, n_hat] = max(abs(C));
                phi_hat    = -angle(C(n_hat));

                err_toa(t) = n_hat - n_true;
                err_phi(t) = wrap_to_pi(phi_hat - phi_true);

                if err_toa(t) == 0
                    n_correct = n_correct + 1;
                end
            end

            rmse_toa(i, k)       = sqrt(mean(err_toa.^2));
            rmse_phi(i, k)       = sqrt(mean(err_phi.^2));
            detection_rate(i, k) = n_correct / N_trames;

                fprintf('     %3d  |  %6.2f  |    %.4f      |     %3.0f %%\n', ...
                    EbN0_dB_range(i), rmse_toa(i,k), rmse_phi(i,k), detection_rate(i,k)*100);
        end
    end

    cmap = lines(length(N_preamble_values));

    figure(fig_base + 2); clf; hold on;
    for k = 1:length(N_preamble_values)
        plot(EbN0_dB_range, rmse_toa(:,k), 'o-', ...
             'Color', cmap(k,:), 'LineWidth', 1.5, ...
             'DisplayName', N_labels{k});
    end
    grid on;
    xlabel('RX SNR (dB)');
    ylabel('RMSE du ToA (samples)');
    title('RMSE du ToA en fonction du SNR');
    legend('Location', 'northeast');
    max_toa = max(rmse_toa(~isnan(rmse_toa)));
    if isempty(max_toa)
        max_toa = 50;
    end
    ylim([0 max(50, 1.1*max_toa)]);

    figure(fig_base + 3); clf; hold on;
    for k = 1:length(N_preamble_values)
        plot(EbN0_dB_range, rmse_phi(:,k), 's-', ...
             'Color', cmap(k,:), 'LineWidth', 1.5, ...
             'DisplayName', N_labels{k});
    end
    grid on;
    xlabel('RX SNR (dB)');
    ylabel('RMSE de phi (rad)');
    title('RMSE du phase offset en fonction du SNR');
    legend('Location', 'northeast');
    ylim([0 2]);

    figure(fig_base + 4); clf; hold on;
    for k = 1:length(N_preamble_values)
        plot(EbN0_dB_range, detection_rate(:,k)*100, 'd-', ...
             'Color', cmap(k,:), 'LineWidth', 1.5, ...
             'DisplayName', N_labels{k});
    end
    yline(100, 'k:', 'LineWidth', 1);
    grid on;
    xlabel('RX SNR (dB)');
    ylabel('Taux de detection correcte (%)');
    title('Probabilite de detection correcte du ToA');
    legend('Location', 'southeast');
    ylim([0 105]);

    fprintf('\n=== Simulation terminee (Step 3) ===\n');
    fprintf('\nINTERPRETATION :\n');
    fprintf('  Figure 4 : Le pic |C| doit etre net a Eb/N0=10dB (peak/floor > 5)\n');
    fprintf('  Figure 5 : RMSE ToA doit chuter brutalement au seuil de detection\n');
    fprintf('             - N=4  : seuil tardif, vers 10-15 dB\n');
    fprintf('             - N=16 : seuil tres tot, vers 0-5 dB\n');
    fprintf('  Figure 6 : RMSE phi decroit doucement (pas de seuil brutal)\n');
    fprintf('  Figure 7 : Taux de detection doit tendre vers 100%% a haut SNR\n');
end

function [eps_hat, syms_sync] = gardner_closed_loop(rx_filt, M, mu, eps_true, nb_symbols, RRC_length)
    L            = length(rx_filt);
    N_max_signal = floor((L - 1) / M) - 1;
    N_symb       = min(nb_symbols, N_max_signal);

    eps_hat    = zeros(N_symb, 1);
    syms_sync  = zeros(N_symb, 1);
    eps_hat(1) = 0;

    M2 = M / 2;
    for n = 2 : N_symb
        eps_corr = eps_hat(n-1);
        delay    = (eps_true - eps_corr) * M;

       % Calculate the group delay of the RRC filter matching block
        % RRC_length is in symbols, M is samples/symbol
        %group_delay = (RRC_length / 2) * M;
        
        % Shift the base time forward so index matches the peak of the pulse
        t_prev = 1  + (n-2)*M       + delay;
        t_mid  = 1  + (n-2)*M + M2  + delay;
        t_curr = 1  + (n-1)*M       + delay;

        if t_prev < 1 || t_curr >= L
            eps_hat(n)   = eps_hat(n-1);
            syms_sync(n) = syms_sync(n-1);
            continue;
        end

        x_prev = lin_interp(rx_filt, t_prev);
        x_mid  = lin_interp(rx_filt, t_mid);
        x_curr = lin_interp(rx_filt, t_curr);

        syms_sync(n) = x_curr;
        e_n = real(x_mid * conj(x_curr - x_prev));
        eps_hat(n) = eps_hat(n-1) + mu * e_n;
    end
end

function y = lin_interp(sig, t)
    % Use a 4-point local window for fast Cubic Spline interpolation
    i_floor = floor(t);
    
    % Boundary protection
    if i_floor < 2 || i_floor > length(sig)-2
        y = sig(i_floor);
        return;
    end
    
    % Grab 4 local points around the target
    idx = (i_floor-1) : (i_floor+2);
    local_sig = sig(idx);
    
    % Perform cubic spline interpolation on this tiny window
    y = interp1(idx, local_sig, t, 'spline');
end

function ang = wrap_to_pi(ang)
    ang = mod(ang + pi, 2*pi) - pi;
end

function h_rrc = generate_rrc_filter(sample_rate, symbol_rate, roll_off, RRC_length)
    sps = round(sample_rate / symbol_rate);
    h_rrc = rcosdesign(roll_off, RRC_length, sps, "sqrt");
    h_rrc = h_rrc(:);
end
