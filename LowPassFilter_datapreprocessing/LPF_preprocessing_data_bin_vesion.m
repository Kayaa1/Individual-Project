load('capacity_data.mat')
%% Using fft to distingulish the frequency
fs = 1; %sampling  fs = 1 ,per cycle
L = length(capacity_B0005);  %length 167
Y_fft=fft(capacity_B0005);  % length 167
f = fs*(-L/2:L/2-1)/L; % per 1 Hz
P2 = abs(Y_fft/L); 
figure;
stem(f,fftshift(P2)) % Shifts the zero to the center
title('Double-Sided Amplitude Spectrum of B0005')
xlabel('Frequency $(\frac{1}{167}$ Hz)', 'Interpreter', 'latex');
ylabel('|P(f)|')
%%
figure;
W_L = 10;
n = 0:1:W_L-1;
hw = transpose(hamming(W_L));
hw=hw/sum(hw);
figure;
stem(hw)
xlabel('samples')
ylabel('Amplititude')
title('Normalized Hamming Window')


figure;
H_w = fft(hw,length(f));
P2_H = abs(H_w);
stem(f,fftshift(P2_H));
xlabel('Frequency(1/167)Hz')
ylabel('Amplititude')
title('Normalized Hamming Window Specturm')

figure;
YW = H_w .* Y_fft ;
stem(f,fftshift(abs(YW)))
yw = ifft(YW);
xlabel('Frequency(1/167)Hz')
ylabel('Amplititude')
title("Smoothed Signal's Specturm")

figure;
plot(yw);
hold on;
plot(capacity_B0005)
xlabel('cycles')
ylabel('Capacity(Ah)')
title("Smoothed Capacity vs Original Capcity")

%%
%% Using fft to distingulish the frequency
fs = 1; %sampling  fs = 1 ,per cycle
L = length(capacity_B0005);  %length 167
Y_fft=fft(capacity_B0005);  % length 167
P2 = abs(Y_fft/L); 
P1 = P2(1:L/2+1); 
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(-L/2:L/2-1)/L; % per 1 Hz
figure;
stem(f,fftshift(P2)) % Shifts the zero to the center
title('Double-Sided Amplitude Spectrum of B0005')
xlabel('Frequency $(\frac{1}{167}$ Hz)', 'Interpreter', 'latex');
ylabel('|P(f)|')
%%
a = fftshift(P2);
LPF_window = 16;
LPF = zeros(1,167);
figure;
LPF(84:84+LPF_window) = 1 
lpf = ifft(LPF)
plot(abs(ifftshift(lpf)))
lpf = abs(ifftshift(lpf))
lpf = lpf(84:167)
Result = conv(capacity_B0005,lpf,"same")
figure;
plot(abs(Result))
%%
LPF(84:84+LPF_window) = 1 
lpf = ifft(LPF)
Result = conv(abs(ifftshift(lpf)),capacity_B0005,"same")
figure;
plot(abs(Result))

