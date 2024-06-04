function rmse_R=joint_test_r(SNR1,f_c1,Nsymbols1,Ncarriers1,delta_f1)
%% parameter setting
SNR=SNR1;
f_c = f_c1; % carrier frequency
Nsymbols = Nsymbols1; %number of OFDM symbols
Ncarriers =Ncarriers1*2;  %number of subcarriers 264*12=3168
delta_f =delta_f1;  % subcarrier spacing

% SNR=15;
% f_c = 24*10^9; % carrier frequency
% Nsymbols = 28; %number of OFDM symbols
% Ncarriers =512;  %number of subcarriers 264*12=3168
% delta_f =120*10^3;  % subcarrier spacing

Nslot = 10;
Nprb = 40;
N_GB = 0;  % number of guard band carriers
Nfft = Ncarriers+2*N_GB; %number of FFT points for OFDM3300
Nfft_d = Ncarriers; Nfft_v = 256; %number of FFT points for distance/velocity estimation for Strum methods
c0 = 3*10^8;
R = [47 51]; %multiple target distance
v = [18 30]; %multiple target velocity

Tofdm = 1/delta_f; %OFDM period (exluding CP)
b=log2(delta_f/15/(10^3));
T=(((1/2)^b)/14)*10^(-3);
% T = 0.125e-3/14; %OFDM period (including CP)
Tcp = T - Tofdm; %CP period
Ncp = round(Nfft*Tcp/Tofdm); %length of CP

%% PRS����
K_comb=2;%comb��ʽΪ4
PRS_carriers=Ncarriers/K_comb;%PRS���ز���
alpha=1;%PRSӳ����������
%% PRSƵ��ӳ��
carriers1=zeros(Nsymbols*(12/14),PRS_carriers);%��ų���PRS�����ز����
for m=1:Nsymbols*(12/14)
    for n=1:PRS_carriers
        carriers1(m,n)=K_comb*(n-1)+(1+(-1)^mod(m,2))/2+1;%����OFDM�������ز������
    end
end
carriers1=carriers1';
%% PRS��������
r1=zeros(Nsymbols*(12/14),PRS_carriers);%PRS����
for m=1:Nsymbols*(12/14)
    x=14*floor((m-1)/12)+(m-12*floor((m-1)/12))+1;%1,2,3,4,5,6,7,8,9,10,11,12,13,��Ӧ2��3��4��5��6��7��8��9��10��11��12��13��16
    n_slot=ceil(x/14)-1;
    c1=goldseq(n_slot,x-1);
    for k=1:PRS_carriers
        r1(m,k)=1/sqrt(2)*(1-2*c1(2*k))+1j*1/sqrt(2)*(1-2*c1(2*k+1)); %�����������PRS����
    end
end
%% DMRS����
K_comb2=4/14;%comb��ʽΪ4/14
DMRS_carriers=Ncarriers/2;%DMRS���ز���
alpha=1;%DMRSӳ����������
%% DMRSƵ��ӳ��
carriers2=zeros(Nsymbols*(4/14),DMRS_carriers);%��ų���DMRS�����ز����
for m=1:Nsymbols*(4/14)
    for n=1:DMRS_carriers
        carriers2(m,n)=K_comb*(n-1)+1;%����OFDM�������ز������
    end
end
carriers2=carriers2';
%% DMRS��������
r2=zeros(Nsymbols*(4/14),DMRS_carriers);%DMRS����
for m=1:Nsymbols*(4/14)
    if mod(m-1,4)<2
        j=1;
    else j=2;
    end
    x=14*floor((m-1)/4)+2*(m-4*floor((m-1)/4))+1+(-1)^j+1;%1��2��3��4��5��6��Ӧ3��5��9��11��17��19 ����
    n_slot=ceil(x/14)-1;
    c2=DMRSgoldseq(n_slot,x-1);
    for k=1:PRS_carriers
        r2(m,k)=1/sqrt(2)*(1-2*c2(2*k))+1j*1/sqrt(2)*(1-2*c2(2*k+1)); %�����������DMRS����
    end
end
r1=r1.';
r2=r2.';
%% �����������ݲ���
% P_data=randi([0 1],1,Ncarriers*Nsymbols*2);%need 3168*20*2 bit
% M=4;
%% qpsk����
% data_temp1= reshape(P_data,log2(M),[])';             %��ÿ��2���ؽ��з��飬M=4
% data_temp2= bi2de(data_temp1);                             %������ת��Ϊʮ���� 3168*20
% modu_data=pskmod(data_temp2,M,pi/M);              % 4PSK����

%% ��Ƶ
% mseq = [0 0 1 1 1 0 1];    % ��Ƶ������� 3168*7
% code = mseq * 2 - 1;         %��1��0�任Ϊ1��-1
%modu_data=reshape(modu_data,Ncarriers,length(modu_data)/Ncarriers);
%spread_data = spread(modu_data,code);        % ��Ƶ 3168*20*7
%spread_data=modu_data;

%% OFDM modulation
IFFT_matrix=zeros(Nfft_d,Nsymbols);
for m=1:Nsymbols*(12/14)
    x=14*floor((m-1)/12)+(m-12*floor((m-1)/12))+1;%1,2,3,4,5,6,7,8,9,10,11,12,13,��Ӧ2��3��4��5��6��7��8��9��10��11��12��13��16
    IFFT_matrix(N_GB+carriers1(:,m),x)=alpha*r1(:,m);
end
for m=1:Nsymbols*(4/14)
    if mod(m-1,4)<2
        j=1;
    else j=2;
    end
    x=14*floor((m-1)/4)+2*(m-4*floor((m-1)/4))+1+(-1)^j+1;%1��2��3��4��5��6��Ӧ3��5��9��11��17��19 ����
    IFFT_matrix(N_GB+carriers2(:,m),x)=alpha*r2(:,m);
end 
IFFT_after=ifft(IFFT_matrix,Nfft_d);

Tx_data=IFFT_after;


%% ��������
% SER_dB=zeros(1,13);
% SNR_dB=-10:5:50;
% for ind=1:13
%     Rx_data1=awgn(Tx_data,SNR_dB(ind));
% %     Rx_data1=Tx_data;
%
%     %% ���
%     Rx_complex_carrier_matrix=fft(Rx_data1,Nfft_d);
%     channelinfo=zeros(Ncarriers,Nsymbols);
%     for k=1:Nsymbols
%         channelinfo(:,k)= Rx_complex_carrier_matrix(N_GB+1:N_GB+Ncarriers,k);
%     end
%     %% despread
%     despreadout=func_despread(channelinfo,code);
%     demod_result=pskdemod(despreadout,M,pi/M);
%     demod_result=reshape(demod_result,Ncarriers*20,1);
%     %% ===============������===============
%     ser=0;
%     for i=1:Ncarriers*20
%         if(demod_result(i,1)==data_temp2(i,1))
%            ser=ser+0;
%         else
%             ser=ser+1;
%         end
%     end
%     SER_dB(1,ind)=ser/(Ncarriers*20);
% end
% figure(1)
% semilogy(SNR_dB,SER_dB);
% title(sprintf('Spread spectrum OFDM SER'))
% ylabel('SER'); xlabel('SNR in dB');



%% ���
Nt=100;
sum_R=0;
sum_V=0;
for t=1:Nt
    Rx_data=Tx_data;
%     Rx_complex_carrier_matrix=Rx_data;
    Rx_complex_carrier_matrix=fft(Rx_data,Nfft_d);
    channelinfo_noise=awgn(Rx_complex_carrier_matrix,SNR);
    channelinfo=zeros(Nfft_d,Nsymbols);
    %*******************����Kr �� Kd ����(����strum������)*******************
    doa=0:1:(Nfft_d-1);
    w=0:1:(Nsymbols-1);
    %*******************���� Kr ����*******************
    kr = zeros(1,Ncarriers);% carrier_count�����ز�����
    % kr = exp(-1j*4*pi*doa.'*delta_f*R(1)/c);
    kr = exp(-1j*4*pi*doa.'*delta_f*R(1)/c0);
    kr_1= zeros(1,Ncarriers);
    kr_1 = exp(-1j*4*pi*doa.'*delta_f*R(2)/c0);
    %*******************���� Kd ����*******************
    kd = zeros(1,Nsymbols);
    % T_OFDM = 1/delta_f * (1 + PrefixRatio);
    % kd = exp(1j * 2 * pi * T_OFDM *m.' * 2 * v * f_c / c);
    kd = exp(1j * 2 * pi * Tofdm *w.'  * 2 * v(1) * f_c / c0);
    kd_1 = zeros(1,Nsymbols);
    kd_1 = exp(1j * 2 * pi * Tofdm *w.' * 2 * v(2) * f_c / c0);
    channelinfo = 1 * channelinfo_noise .* (kr *  kd');
    channelinfo_two = 1 * channelinfo_noise .* (kr_1 *  kd_1');
    channelinfo_twosignal= channelinfo ;%��Ŀ��
    % Rx_complex_carrier_matrix_radar = Rx_complex_carrier_matrix_radar./complex_carrier_matrix;
    % Rx_complex_carrier_matrix_radar = Rx_complex_carrier_matrix_radar';
    
    % channelinfo=zeros(PRS_carriers,Nsymbols);
    % for k=1:Nsymbols
    %     channelinfo(:,k)= Rx_complex_carrier_matrix1(N_GB+carriers(:,k),k);
    % end
    process=1;
    switch process
        case 1% ==========2D FFT====================
            %% �����ŵ���Ϣ����
            %% ��ȡ���ڲ��ĵ�Ƶ��Ϣ����
            Rx_complex_carrier_matrix_1=zeros(Ncarriers,Nsymbols*(4/14));
            for k=1:Nsymbols*(4/14)
                if mod(k-1,4)<2
                    j=1;
                else j=2;
                end
                x=14*floor((k-1)/4)+2*(k-4*floor((k-1)/4))+1+(-1)^j+1;%1��2��3��4��5��6��Ӧ3��5��9��11��17��19 ����
                Rx_complex_carrier_matrix_1(:,k)= Rx_complex_carrier_matrix(:,x);
            end
            channelinfo_1=zeros(Ncarriers,Nsymbols*(4/14));
            for k=1:Nsymbols*(4/14)
                if mod(k-1,4)<2
                    j=1;
                else j=2;
                end
                x=14*floor((k-1)/4)+2*(k-4*floor((k-1)/4))+1+(-1)^j+1;%1��2��3��4��5��6��Ӧ3��5��9��11��17��19 ����
                channelinfo_1(:,k)= channelinfo(:,x);
            end
            %% ��ȡ���ڲ��ٵĵ�Ƶ��Ϣ����
            %                     symbols=zeros(Ncarriers,Nsymbols/K_comb);%��ų���PRS�����ز����
            %                     for m=1:Ncarriers
            %                        for n=1:Nsymbols/K_comb
            %                            symbols(m,n)=K_comb*(n-1)+(1+(-1)^mod(m,2))/2+1;%����OFDM���ŵ����
            %                        end
            %                     end
            %                     Rx_complex_carrier_matrix_2=zeros(Ncarriers,Nsymbols/K_comb);
            %                     for k=1:Ncarriers
            %                           Rx_complex_carrier_matrix_2(k,:)= Rx_complex_carrier_matrix(N_GB+k,symbols(k,:));
            %                     end
            %                     channelinfo_2=zeros(Ncarriers,Nsymbols/K_comb);
            %                     for k=1:Ncarriers
            %                          channelinfo_2(k,:)= channelinfo(N_GB+k,symbols(k,:));
            %                     end
            %         kr = zeros(1,Ncarriers);
            %         for k = 1:Ncarriers
            %             kr(k) = exp(-1i * 2 * pi * (k-1) * delta_f * 2 * R(1) / c);
            %         end
            %
            %         kd = zeros(1,Nsymbols);
            %         for k = 1:Nsymbols
            %             kd(k) = exp(1i * 2 * pi * Tofdm * (k-1) * 2 * v(1) * f_c / c);
            %         end
            %         Rx_complex_carrier_matrix_radar = 1 * channelinfo .* (kr' *  kd);
            %         Rx_complex_carrier_matrix_radar1=awgn(Rx_complex_carrier_matrix_radar,30);
            %% ======================���===================================
            index_r=0;%��������
            div_IFFT=zeros(1,Ncarriers);%��IFFTѰ�ҷ�ֵʱ������IFFT�Ľ��
            for i=1:Nsymbols*(4/14)
                div_R = channelinfo_1 (:,i) ./ Rx_complex_carrier_matrix_1(:,i);
                cur_div_IFFT = ifft(div_R,Nfft_d,1);
                [max_R,index_R] = max(cur_div_IFFT);
                index_r = index_r + index_R;
                div_IFFT = div_IFFT + cur_div_IFFT;
            end
            index_r = index_r / (Nsymbols*(4/14));
            M_R = ((index_r-1) / Ncarriers) * (c0 / 2 / delta_f);
            %���
            figure(2);
            div_IFFT = div_IFFT / (Nsymbols*(4/14));
            Index_r=[0:Ncarriers-1];
            R_=[1:Ncarriers];
            R_=(Index_r'./ Ncarriers) * (c0 / 2 / delta_f);
            plot(R_,abs(div_IFFT));
%             plot(abs(div_IFFT));
%             text(index_r,abs(div_IFFT(floor(index_r))),'o','color','r');  %��ǳ���ֵ��
%             text(floor(index_r)+20,abs(div_IFFT(round(index_r))),['(',num2str(floor(index_r)),',',num2str(abs(div_IFFT(round(index_r)))),')'],'color','k');  %��ǳ���ֵ��
            xlabel('Distance/m');
            ylabel('Mag');
            
            sum_R=sum_R+(M_R-R(1))^2;
            %% ======================����===================================
            %                     index_v=0;%���ٵ�����
            %                     div_FFT=zeros(1,Nsymbols/K_comb);%��FFTѰ�ҷ�ֵʱ������FFT�Ľ��
            %                     for i=1:Ncarriers
            %                         div_v = (channelinfo_2 (i,:) ./ Rx_complex_carrier_matrix_2(i,:));
            %                         cur_div_FFT = ifft(div_v);
            %                         [max_v,cur_index_v] = max(cur_div_FFT);
            %                         index_v = index_v + cur_index_v;
            %                         div_FFT = div_FFT + cur_div_FFT;
            %                     end
            %
            %                     index_v = index_v / Ncarriers;
            %                     % ���������еĹ�ʽ��������ٶ�
            %                     M_v = ((index_v-1)*c0 / (2*f_c*Nsymbols*(1/delta_f)));
            %                     %����
            %                     figure(3)
            %                     div_FFT = div_FFT / Ncarriers;
            %                     plot(abs(div_FFT));
            %                     text(floor(index_v),abs(div_FFT(round(index_v))),'o','color','r');  %��ǳ���ֵ��
            %                     xlabel('index_v');
            %                     ylabel('Mag');
            %                     title('OFDM radar velocity');
            %                     sum_V=sum_V+(M_v-v(1))^2;
        case 2%MUSIC
            % �����ŵ���Ϣ����
            %% ��ȡ���ڲ��ĵ�Ƶ��Ϣ����
            Rx_complex_carrier_matrix_1=zeros(PRS_carriers,Nsymbols);
            for k=1:Nsymbols
                Rx_complex_carrier_matrix_1(:,k)= Rx_complex_carrier_matrix(N_GB+carriers(:,k),k);
            end
            channelinfo_1=zeros(PRS_carriers,Nsymbols);
            for k=1:Nsymbols
                channelinfo_1(:,k)= channelinfo_twosignal(N_GB+carriers(:,k),k);
            end
            %% ��ȡ���ڲ��ٵĵ�Ƶ��Ϣ����
            symbols=zeros(Ncarriers,Nsymbols/K_comb);%��ų���PRS�ķ��ŵ����
            for m=1:Ncarriers
                for n=1:Nsymbols/K_comb
                    symbols(m,n)=K_comb*(n-1)+mod(m-1,K_comb)/2+3/4*(1-(-1)^mod(m-1,K_comb))+1;%����OFDM���ŵ����
                end
            end
            Rx_complex_carrier_matrix_2=zeros(Ncarriers,Nsymbols/K_comb);
            for k=1:Ncarriers
                Rx_complex_carrier_matrix_2(k,:)= Rx_complex_carrier_matrix(N_GB+k,symbols(k,:));
            end
            channelinfo_2=zeros(Ncarriers,Nsymbols/K_comb);
            for k=1:Ncarriers
                channelinfo_2(k,:)= channelinfo_twosignal(N_GB+k,symbols(k,:));
            end
            %% *******************���վ����뷢�;������*******************
            Rx_complex_carrier_matrix_radar_1=channelinfo_1./Rx_complex_carrier_matrix_1;%���ڲ��ĵ�Ƶ����
            Rx_complex_carrier_matrix_radar_2=channelinfo_2./Rx_complex_carrier_matrix_2;%���ڲ��ٵĵ�Ƶ����
            %% ********MUSIC�㷨-distance********
            Ryy=zeros(PRS_carriers,PRS_carriers);
            for i=1:Nsymbols
                Ry_distance = Rx_complex_carrier_matrix_radar_1(:,i); % ��ȡ��һ����ʱ�ӹ���
                Ryy=Ryy+Ry_distance*Ry_distance'; % ����������ؾ��󣨾����������ת�ã�Ϊ����ؾ���
            end
            Ryy=Ryy/Nsymbols;
            %         Ry_distance = Rx_complex_carrier_matrix_radar1(:,1); % ��ȡ��һ����ʱ�ӹ���
            %         Ryy=Ry_distance*Ry_distance'; % ����������ؾ��󣨾����������ת�ã�Ϊ����ؾ���
            [U,S,V]=svd(Ryy);   % ����ֽ�
            G=U(:,2:PRS_carriers);  % �����ӿռ�
            Gn=G*G';
            search_r=0:0.03:100;
            doa=0:1:(PRS_carriers-1);
            for i=1:length(search_r)
                r=exp(-1j * 2 * pi * K_comb * doa' * delta_f * 2 * search_r(i) / c0);
                p(i)=1./abs(r'*Gn*r);
            end
            [max_r,index_r]=max(p);
            figure(3);
            plot(search_r,10*log(p),'r');
            disp((index_r-1)*0.03);
            xlabel('����/(m)');
            ylabel('�ռ���/(dB)');
            title('MUSIC�㷨���ͼ');
            M_R=(index_r-1)*0.03;
            sum_R=sum_R+(M_R-R(1))^2;
            %% ********MUSIC�㷨-velocity********
            Ryy2=zeros(Nsymbols/K_comb,Nsymbols/K_comb);
            for i=1:Ncarriers
                Ry_velocity = Rx_complex_carrier_matrix_radar_2(i,:); % ��ȡ��һ����ʱ�ӹ���
                Ryy2=Ryy2+Ry_velocity'*Ry_velocity; % ����������ؾ��󣨾����������ת�ã�Ϊ����ؾ���
            end
            Ryy2=Ryy2/Ncarriers;
            [U2,S2,V2]=svd(Ryy2);   % ����ֽ�
            G2=U2(:,2:(Nsymbols/K_comb));  % �����ӿռ�
            Gn2=G2*G2';
            search_v=0:0.1:100;
            m=0:1:(Nsymbols/K_comb-1);
            for i=1:length(search_v)
                r_velocity=exp(1j * 2 * pi * K_comb * Tofdm *m' * 2 * search_v(i) * f_c / c0);
                p2(i)=1./abs(r_velocity'*Gn2*r_velocity);
            end
            [max_v,index_v]=max(p2);
            figure(4);
            plot(search_v,10*log(p2),'b');
            disp(index_v*0.1);
            xlabel('�ٶ�/(m/s)');
            ylabel('�ռ���/(dB)');
            title('MUSIC�㷨����ͼ');
            M_v=index_v*0.1;
            sum_V=sum_V+(M_v-v(1))^2;
    end
end
rmse_R=sqrt(sum_R/Nt)
%         rmse_V=sqrt(sum_V/Nt)
 end