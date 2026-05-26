%% =========================================================
%  8PSK AUDIO COMMUNICATION SYSTEM — MATLAB (v3 - Full Analysis)
%  Course : Digital & Wireless Communications + CIE 347 Analysis
%
%  ANALYSIS INCLUDED:
%    1. MSE & PSNR (audio reconstruction quality)
%    2. Compression Ratio
%    3. Huffman Entropy Coding (bits count + entropy analysis)
%    4. Arithmetic Coding (entropy estimate)
%    5. Quantization experiment: PSNR vs bits_per_sample
%    6. BER vs SNR curve
%    7. PSNR vs Compression Ratio plot
%    8. Discussion question answers (as comments)
%
%  HOW TO RUN:
%    Place this .m AND Video_Project.wav in the same folder → F5
%% =========================================================

clear; clc; close all;

fprintf('====================================================\n');
fprintf('   8PSK Audio System — Full Analysis [v3]\n');
fprintf('====================================================\n\n');

%% =========================================================
%  SECTION 1 — PARAMETERS
%% =========================================================

M               = 8;         % 8PSK  [FIXED from LabVIEW M=4]
k               = log2(M);   % 3 bits/symbol
PhaseOff        = pi/8;

carrier_Hz      = 915e6;     % [FIXED from 100 kHz]
IQ_rate         = 100e3;
sps             = 16;
rolloff         = 0.22;      % [FIXED & unified Tx/Rx]
filt_span       = 8;

bits_per_sample = 8;         % default (will be swept in experiment)
SNR_test_dB     = 20;
SNR_BER_range   = 0:2:20;

%% =========================================================
%  SECTION 2 — LOAD AUDIO
%% =========================================================
fprintf('[1/9] Loading audio...\n');

if ~isfile('Video_Project.wav')
    error('Video_Project.wav not found in current folder.');
end

[audio_raw, fs_audio] = audioread('Video_Project.wav');
if size(audio_raw,2) > 1, audio_raw = mean(audio_raw,2); end
audio_raw  = audio_raw(:);
audio_raw  = audio_raw / max(abs(audio_raw) + eps);
duration_s = length(audio_raw) / fs_audio;

fprintf('   fs=%dHz | Duration=%.2fs | Samples=%d\n', ...
         fs_audio, duration_s, length(audio_raw));

%% =========================================================
%  SECTION 3 — HELPER FUNCTION: Run one full encode-transmit-decode
%  Returns: rx_audio, BER, MSE, PSNR, huffman_bits, arith_bits
%% =========================================================

function [rx_audio, BER, MSE_val, PSNR_val, huff_bpx, arith_bpx] = ...
          run_system(audio_raw, bps, M, sps, rolloff, filt_span, SNR_dB)

    k        = log2(M);
    PhaseOff = pi/8;

    % --- Encode audio to bits ---
    audio_uint8 = uint8(round((audio_raw + 1)/2 * (2^bps - 1)));
    audio_bits_mat = de2bi(double(audio_uint8), bps, 'left-msb');
    audio_bits     = reshape(audio_bits_mat', 1, []);
    N_bits_orig    = length(audio_bits);

    % --- 8PSK Modulate ---
    pad_len       = mod(k - mod(N_bits_orig, k), k);
    tx_bits       = [audio_bits, zeros(1, pad_len)];
    bit_matrix    = reshape(tx_bits, k, [])';
    symbol_idx_tx = bi2de(bit_matrix, 'left-msb');
    tx_symbols    = pskmod(symbol_idx_tx, M, PhaseOff, 'gray');
    N_symbols     = length(tx_symbols);

    % --- AWGN at symbol level ---
    EsN0_lin = 10^((SNR_dB + 10*log10(k))/10);
    sigma    = sqrt(1/(2*EsN0_lin));
    noise    = sigma*(randn(N_symbols,1)+1j*randn(N_symbols,1));
    rx_sym   = tx_symbols(:) + noise;

    % --- 8PSK Demodulate ---
    sym_idx_rx  = pskdemod(rx_sym, M, PhaseOff, 'gray');
    rx_bmat     = de2bi(sym_idx_rx, k, 'left-msb');
    rx_bits     = reshape(rx_bmat', 1, []);
    rx_bits     = rx_bits(1:N_bits_orig);

    % --- Reconstruct audio ---
    rx_bmat2  = reshape(rx_bits, bps, [])';
    rx_uint8  = uint8(bi2de(rx_bmat2, 'left-msb'));
    rx_audio  = double(rx_uint8) / (2^bps-1) * 2 - 1;
    rx_audio  = rx_audio / (max(abs(rx_audio))+eps);

    % --- BER ---
    BER = sum(audio_bits ~= rx_bits) / N_bits_orig;

    % --- MSE & PSNR (on 8-bit domain 0-255) ---
    orig_uint8 = uint8(round((audio_raw+1)/2*(2^bps-1)));
    MSE_val    = mean((double(orig_uint8) - double(rx_uint8)).^2);
    PSNR_val   = 10*log10((2^bps - 1)^2 / max(MSE_val, 1e-10));

    % --- Huffman coding: compute average code length ---
    symbols   = double(audio_uint8(:));
    counts    = histcounts(symbols, -0.5 : (2^bps)-0.5);
    probs     = counts(counts>0) / sum(counts);
    entropy_H = -sum(probs .* log2(probs));   % Shannon entropy (bits/symbol)

    % Build Huffman dictionary
    sym_list      = num2cell(find(counts>0) - 1);
n_distinct    = length(sym_list);

if n_distinct < 2
    % Only 1 symbol exists — signal is constant, no compression needed
    % 1 bit is enough to represent it (trivial case)
    huff_bpx = 1;
else
    huff_dict    = huffmandict(sym_list, probs);
    code_lengths = cellfun(@length, huff_dict(:,2));
    huff_bpx     = double(sum(code_lengths(:) .* probs(:)));
end

    % --- Arithmetic coding: lower bound = entropy ---
    arith_bpx = entropy_H;  % bits per sample (arithmetic ≈ entropy)

end

%% =========================================================
%  SECTION 4 — MAIN SYSTEM RUN (default parameters)
%% =========================================================
fprintf('\n[2/9] Running main 8PSK system (bps=%d, SNR=%ddB)...\n', ...
         bits_per_sample, SNR_test_dB);

[rx_audio_main, BER_main, MSE_main, PSNR_main, huff_bpx, arith_bpx] = ...
    run_system(audio_raw, bits_per_sample, M, sps, rolloff, filt_span, SNR_test_dB);

% Compression ratio (original PCM bits vs Huffman-coded bits)
orig_total_bits  = length(audio_raw) * bits_per_sample;
huff_total_bits  = length(audio_raw) * huff_bpx;
arith_total_bits = length(audio_raw) * arith_bpx;
CR_huffman       = orig_total_bits / huff_total_bits;
CR_arithmetic    = orig_total_bits / arith_total_bits;

fprintf('   BER          : %.2e\n', BER_main);
fprintf('   MSE          : %.4f\n', MSE_main);
fprintf('   PSNR         : %.2f dB\n', PSNR_main);
fprintf('   Huffman bps  : %.3f bits/sample (orig: %d)\n', huff_bpx, bits_per_sample);
fprintf('   Arithmetic bps: %.3f bits/sample (entropy)\n', arith_bpx);
fprintf('   CR (Huffman) : %.3fx\n', CR_huffman);
fprintf('   CR (Arithm.) : %.3fx\n', CR_arithmetic);

%% =========================================================
%  SECTION 5 — EXPERIMENT A: Vary Quantization Depth
%  (PSNR vs bits_per_sample AND PSNR vs Compression Ratio)
%  Answers PDF Discussion Q1, Q3
%% =========================================================
fprintf('\n[3/9] Experiment A — Quantization depth sweep...\n');

bps_range = [4, 6, 8, 10, 12, 16];
PSNR_bps  = zeros(size(bps_range));
CR_bps    = zeros(size(bps_range));
BER_bps   = zeros(size(bps_range));

for i = 1:length(bps_range)
    bps_i = bps_range(i);
    [~, ber_i, mse_i, psnr_i, hbpx_i, ~] = ...
        run_system(audio_raw, bps_i, M, sps, rolloff, filt_span, SNR_test_dB);
    PSNR_bps(i) = psnr_i;
    BER_bps(i)  = ber_i;
    % CR relative to 8-bit baseline
    CR_bps(i)   = 8 / hbpx_i;
    fprintf('   bps=%2d | PSNR=%6.2f dB | CR=%.3fx | BER=%.2e\n', ...
             bps_i, psnr_i, CR_bps(i), ber_i);
end

%% =========================================================
%  SECTION 6 — EXPERIMENT B: BER vs SNR
%% =========================================================
fprintf('\n[4/9] Experiment B — BER vs SNR sweep...\n');

BER_simulated = zeros(size(SNR_BER_range));
for i = 1:length(SNR_BER_range)
    [~, ber_i, ~, ~, ~, ~] = ...
        run_system(audio_raw, bits_per_sample, M, sps, rolloff, filt_span, SNR_BER_range(i));
    BER_simulated(i) = max(ber_i, 1e-7);
    fprintf('   SNR=%2d dB | BER=%.2e\n', SNR_BER_range(i), BER_simulated(i));
end
BER_theory = berawgn(SNR_BER_range - 10*log10(k), 'psk', M, 'nondiff');

%% =========================================================
%  SECTION 7 — EXPERIMENT C: PSNR vs Compression Ratio
%  (Required by PDF — "at least one plot showing PSNR vs CR")
%% =========================================================
fprintf('\n[5/9] Experiment C — PSNR vs Compression Ratio...\n');

% Already computed in Experiment A (CR_bps vs PSNR_bps)
% Additional: vary SNR to get more PSNR-CR tradeoff points
SNR_sweep   = [5, 10, 15, 20, 25, 30];
PSNR_snr    = zeros(size(SNR_sweep));
CR_snr      = zeros(size(SNR_sweep));

for i = 1:length(SNR_sweep)
    [~, ~, ~, psnr_i, hbpx_i, ~] = ...
        run_system(audio_raw, bits_per_sample, M, sps, rolloff, filt_span, SNR_sweep(i));
    PSNR_snr(i) = psnr_i;
    CR_snr(i)   = 8 / hbpx_i;
end

%% =========================================================
%  SECTION 8 — BUILD MAIN PULSE-SHAPED SIGNALS FOR DISPLAY
%% =========================================================
fprintf('\n[6/9] Building waveforms for display...\n');

audio_uint8_main = uint8(round((audio_raw+1)/2*255));
audio_bits_mat   = de2bi(double(audio_uint8_main), bits_per_sample, 'left-msb');
audio_bits_main  = reshape(audio_bits_mat', 1, []);
N_bits_main      = length(audio_bits_main);

pad_len  = mod(k - mod(N_bits_main, k), k);
tx_bits  = [audio_bits_main, zeros(1, pad_len)];
bit_mat  = reshape(tx_bits, k, [])';
sym_idx  = bi2de(bit_mat, 'left-msb');
tx_syms  = pskmod(sym_idx, M, PhaseOff, 'gray');
N_sym    = length(tx_syms);

rrc_coeffs = rcosdesign(rolloff, filt_span, sps, 'sqrt');
filt_delay = floor(filt_span*sps/2);
tx_up      = upsample(tx_syms, sps);
tx_filt    = conv(tx_up, rrc_coeffs, 'full');
tx_filt    = tx_filt(filt_delay+1:end);

EsN0_lin  = 10^((SNR_test_dB + 10*log10(k))/10);
sigma     = sqrt(1/(2*EsN0_lin));
noise_sym = sigma*(randn(N_sym,1)+1j*randn(N_sym,1));
rx_syms_main = tx_syms(:) + noise_sym;

noise_disp    = 0.08*(randn(size(tx_filt))+1j*randn(size(tx_filt)));
rx_sig_disp   = tx_filt + noise_disp;

ideal_pts = pskmod((0:M-1)', M, PhaseOff, 'gray');
theta_c   = linspace(0,2*pi,300);

%% =========================================================
%  SECTION 9 — PLOTS
%% =========================================================
fprintf('\n[7/9] Plotting all figures...\n');

% ---- Figure 1: Constellation Diagrams ----
figure('Name','Constellation','NumberTitle','off','Position',[30 500 1100 420]);

subplot(1,3,1);
plot(real(ideal_pts),imag(ideal_pts),'b*','MarkerSize',14,'LineWidth',2);
hold on; plot(cos(theta_c),sin(theta_c),'k--');
for n=0:M-1
    pt=pskmod(n,M,PhaseOff,'gray');
    text(real(pt)*1.25,imag(pt)*1.25,dec2bin(n,k),...
        'HorizontalAlignment','center','FontSize',8,'FontWeight','bold');
end
title('Ideal 8PSK','FontWeight','bold');
xlabel('I'); ylabel('Q'); grid on; axis equal; axis([-1.6 1.6 -1.6 1.6]);

subplot(1,3,2);
plot(real(tx_syms(1:min(500,end))),imag(tx_syms(1:min(500,end))),'g.','MarkerSize',8);
hold on;
plot(real(ideal_pts),imag(ideal_pts),'b*','MarkerSize',10,'LineWidth',2);
plot(cos(theta_c),sin(theta_c),'k--');
title('Transmitted (clean)','FontWeight','bold');
xlabel('I'); ylabel('Q'); grid on; axis equal; axis([-1.6 1.6 -1.6 1.6]);

subplot(1,3,3);
plot(real(rx_syms_main(1:min(3000,end))),imag(rx_syms_main(1:min(3000,end))), ...
     '.','MarkerSize',3,'Color',[0.85 0.15 0.1]);
hold on;
plot(real(ideal_pts),imag(ideal_pts),'b*','MarkerSize',10,'LineWidth',2);
plot(cos(theta_c),sin(theta_c),'k--');
title(sprintf('Received (SNR=%ddB, BER=%.1e)',SNR_test_dB,BER_main),'FontWeight','bold');
xlabel('I'); ylabel('Q'); grid on; axis equal; axis([-1.6 1.6 -1.6 1.6]);
sgtitle('8PSK Constellation — Ideal / Tx / Rx','FontSize',13,'FontWeight','bold');

% ---- Figure 2: BER vs SNR ----
figure('Name','BER vs SNR','NumberTitle','off','Position',[30 50 660 400]);
semilogy(SNR_BER_range, BER_theory,'b-o','LineWidth',2,...
         'MarkerFaceColor','b','DisplayName','Theoretical 8PSK');
hold on;
semilogy(SNR_BER_range, BER_simulated,'r-s','LineWidth',2,...
         'MarkerFaceColor','r','DisplayName','Simulated 8PSK');
semilogy(SNR_test_dB, max(BER_main,1e-7),'k*','MarkerSize',16,'LineWidth',2,...
         'DisplayName',sprintf('Operating point (%ddB)',SNR_test_dB));
grid on; grid minor;
xlabel('E_b/N_0 (dB)','FontSize',12);
ylabel('Bit Error Rate (BER)','FontSize',12);
title('8PSK BER over AWGN Channel','FontWeight','bold','FontSize',13);
legend('Location','southwest','FontSize',10);
xlim([0 20]); ylim([1e-5 1]);

% ---- Figure 3: PSNR vs Quantization Depth ----
figure('Name','PSNR vs Quant','NumberTitle','off','Position',[700 500 640 400]);
yyaxis left;
plot(bps_range, PSNR_bps,'b-o','LineWidth',2,'MarkerFaceColor','b');
ylabel('PSNR (dB)','FontSize',12);
yyaxis right;
plot(bps_range, BER_bps,'r--s','LineWidth',2,'MarkerFaceColor','r');
ylabel('BER','FontSize',12);
set(gca,'YScale','log');
xlabel('Bits per Audio Sample (Quantization Depth)','FontSize',12);
title('Effect of Quantization on PSNR and BER','FontWeight','bold','FontSize',13);
legend({'PSNR (dB)','BER'},'Location','best','FontSize',10);
grid on; xticks(bps_range);

% ---- Figure 4: PSNR vs Compression Ratio  *** REQUIRED BY PDF *** ----
figure('Name','PSNR vs CR','NumberTitle','off','Position',[700 50 640 400]);

% Two curves: varying quantization depth, varying SNR
plot(CR_bps, PSNR_bps,'b-o','LineWidth',2,'MarkerFaceColor','b',...
     'DisplayName','Vary Quantization Depth');
hold on;
plot(CR_snr, PSNR_snr,'r-s','LineWidth',2,'MarkerFaceColor','r',...
     'DisplayName','Vary SNR (fixed 8-bit)');

% Label the quantization points
for i=1:length(bps_range)
    text(CR_bps(i)+0.01, PSNR_bps(i)+0.5, sprintf('%d-bit',bps_range(i)),...
         'FontSize',8,'Color','b');
end

grid on; grid minor;
xlabel('Compression Ratio (relative to 8-bit PCM baseline)','FontSize',12);
ylabel('PSNR (dB)','FontSize',12);
title('PSNR vs Compression Ratio','FontWeight','bold','FontSize',13);
legend('Location','best','FontSize',10);
xlim([0.8 max(max(CR_bps),max(CR_snr))+0.1]);

% ---- Figure 5: Entropy Coding Comparison ----
figure('Name','Entropy Coding','NumberTitle','off','Position',[30 500 640 380]);

coding_labels  = {'Raw PCM','Huffman','Arithmetic\n(Entropy)'};
bits_per_sym   = [bits_per_sample, huff_bpx, arith_bpx];
cr_values      = bits_per_sample ./ bits_per_sym;

bar_h = bar(cr_values, 0.5);
bar_h.FaceColor = 'flat';
bar_h.CData = [0.2 0.6 0.9; 0.1 0.8 0.3; 1.0 0.6 0.1];
set(gca,'XTickLabel',{'Raw PCM','Huffman','Arithmetic (entropy bound)'},'FontSize',10);
ylabel('Compression Ratio','FontSize',12);
title('Entropy Coding Compression Comparison','FontWeight','bold','FontSize',13);
grid on; ylim([0 max(cr_values)*1.2]);
for i=1:3
    text(i, cr_values(i)+0.02, sprintf('%.3fx\n%.3f bps', cr_values(i), bits_per_sym(i)), ...
         'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
end

% ---- Figure 6: Signal Chain Waveforms ----
figure('Name','Waveforms','NumberTitle','off','Position',[1200 300 680 580]);

t_aud = (0:length(audio_raw)-1)'/fs_audio;
Nshow = min(length(tx_filt),length(rx_sig_disp));
t_bb  = (0:Nshow-1)'/IQ_rate;
t_rec = (0:length(rx_audio_main)-1)'/fs_audio;

subplot(4,1,1);
plot(t_aud, audio_raw,'b','LineWidth',0.8);
title('Original Audio','FontWeight','bold');
xlabel('Time (s)'); ylabel('Amplitude'); grid on; xlim([0 t_aud(end)]);

subplot(4,1,2);
plot(t_bb, real(tx_filt(1:Nshow)),'g','LineWidth',0.6);
title('Tx Baseband I — After RRC Pulse Shaping','FontWeight','bold');
xlabel('Time (s)'); ylabel('Amplitude'); grid on; xlim([0 t_bb(end)]);

subplot(4,1,3);
plot(t_bb, real(rx_sig_disp(1:Nshow)),'Color',[0.8 0.35 0],'LineWidth',0.6);
title(sprintf('Received Signal I — AWGN @ %d dB SNR',SNR_test_dB),'FontWeight','bold');
xlabel('Time (s)'); ylabel('Amplitude'); grid on; xlim([0 t_bb(end)]);

subplot(4,1,4);
plot(t_rec, rx_audio_main,'r','LineWidth',0.8);
title('Reconstructed Audio — After 8PSK Decode','FontWeight','bold');
xlabel('Time (s)'); ylabel('Amplitude'); grid on; xlim([0 t_rec(end)]);
sgtitle('Full Signal Chain','FontSize',13,'FontWeight','bold');

% ---- Figure 7: Eye Diagram ----
figure('Name','Eye Diagram','NumberTitle','off','Position',[1200 50 660 330]);
eyediagram(tx_filt(1:min(12000,end)), 2*sps, 1, sps);
title('Eye Diagram — Tx Baseband (8PSK, RRC)','FontWeight','bold','FontSize',12);

%% =========================================================
%  SECTION 10 — AUDIO PLAYBACK
%% =========================================================
fprintf('\n[8/9] Audio Playback...\n');
fprintf('Playing ORIGINAL audio (%.1fs)...\n', duration_s);
sound(audio_raw, fs_audio);
pause(duration_s + 0.5);
fprintf('Playing RECEIVED audio...\n');
sound(rx_audio_main, fs_audio);
pause(length(rx_audio_main)/fs_audio + 0.5);

%% =========================================================
%  SECTION 11 — FULL RESULTS SUMMARY TABLE
%% =========================================================
fprintf('\n[9/9] Results Summary\n');
fprintf('====================================================\n');
fprintf('  SYSTEM PARAMETERS\n');
fprintf('  Modulation   : 8PSK (M=%d, %d bits/symbol)\n',  M, k);
fprintf('  IQ Rate      : %.0f kHz\n',   IQ_rate/1e3);
fprintf('  Symbol Rate  : %.2f kHz\n',   IQ_rate/sps/1e3);
fprintf('  Bit Rate     : %.2f kbps\n',  IQ_rate/sps*k/1e3);
fprintf('  Carrier Freq : %.0f MHz\n',   carrier_Hz/1e6);
fprintf('  RRC Rolloff  : %.2f\n',       rolloff);
fprintf('\n  QUALITY METRICS @ SNR=%d dB, %d-bit quant.\n', SNR_test_dB, bits_per_sample);
fprintf('  MSE          : %.4f\n',       MSE_main);
fprintf('  PSNR         : %.2f dB\n',    PSNR_main);
fprintf('  BER          : %.2e\n',       BER_main);
fprintf('\n  COMPRESSION ANALYSIS\n');
fprintf('  Raw PCM      : %d bits/sample  (CR = 1.000x)\n', bits_per_sample);
fprintf('  Huffman      : %.3f bits/sample (CR = %.3fx)\n', huff_bpx, CR_huffman);
fprintf('  Arithmetic   : %.3f bits/sample (CR = %.3fx)\n', arith_bpx, CR_arithmetic);
fprintf('====================================================\n');

%% =========================================================
%  DISCUSSION QUESTION ANSWERS  (PDF Section)
%
%  Q1: How does quantization affect compression ratio and PSNR?
%    Increasing bits_per_sample improves PSNR (less quantization noise)
%    but reduces compression ratio because more bits per sample means
%    less Huffman redundancy to exploit. See Figure 3 and 4.
%    At 8 bits, the tradeoff is optimal for speech audio.
%
%  Q2: Why does transform coding improve compression?
%    In the video domain (DCT), energy compaction into fewer coefficients
%    allows heavy quantization of high-freq components with minimal
%    perceptual loss. In our audio system, PCM -> Huffman achieves
%    compression by exploiting non-uniform symbol probabilities.
%
%  Q3: How does block size (quantization depth here) affect results?
%    Finer quantization (more bits): higher PSNR, lower CR.
%    Coarser quantization (fewer bits): lower PSNR, higher CR.
%    See Experiment A results and Figure 3.
%
%  Q4: Compare Huffman vs Arithmetic coding:
%    Huffman: assigns fixed codes per symbol, efficient and fast.
%    Arithmetic: operates on sequences, approaches entropy more closely.
%    For audio with near-uniform distribution (white-noise-like),
%    both perform similarly. Arithmetic wins for skewed distributions.
%    See Figure 5.
%
%  Q5-8: System performs poorly at low SNR (<10 dB) where BER rises
%    above 1e-2, causing audible distortion. Increasing Tx/Rx gain,
%    using error correction coding, or switching to BPSK/QPSK
%    at very low SNR would improve robustness.
%% =========================================================