%测距测速CRLB关于符号长度的权衡考虑
% f_c2=24*10^9;
% ma1=1;
% symbols_per_carrier1=140;
% carrier_count1=128; 
% symbols=symbols_per_carrier1*(4/14);
% carrier_count=carrier_count1;
% f_c=f_c2;
% c = 3*10^8;
% SNR=0;
% linear_SNR=10^(SNR/10);
% for i=1:5
%          delta_f1=15*10^3*(2^(5-i));
%          delta_f=delta_f1;
%          
%          a=0.3;
% CRBD_3(i)=sqrt(12*c^2/delta_f^2/(4*pi^2)/0.5^2/linear_SNR/(carrier_count*2)/symbols/((carrier_count*2)-1)/(7*(carrier_count*2)));
% CRBV_3(i)=sqrt(12*c^2*delta_f^2/f_c^2/(4*pi^2)/0.5^2/linear_SNR/carrier_count/symbols_per_carrier1/(symbols_per_carrier1-1)/(7*symbols_per_carrier1));
% CRB_Tar(i)=a*CRBD_3(i)+(1-a)*CRBV_3(i);
%         a=0.4;
% CRBD_3(i)=sqrt(12*c^2/delta_f^2/(4*pi^2)/0.5^2/linear_SNR/(carrier_count*2)/symbols/((carrier_count*2)-1)/(7*(carrier_count*2)));
% CRBV_3(i)=sqrt(12*c^2*delta_f^2/f_c^2/(4*pi^2)/0.5^2/linear_SNR/carrier_count/symbols_per_carrier1/(symbols_per_carrier1-1)/(7*symbols_per_carrier1));
% CRB_Tar2(i)=a*CRBD_3(i)+(1-a)*CRBV_3(i);
%         a=0.5;  
% CRBD_3(i)=sqrt(12*c^2/delta_f^2/(4*pi^2)/0.5^2/linear_SNR/(carrier_count*2)/symbols/((carrier_count*2)-1)/(7*(carrier_count*2)));
% CRBV_3(i)=sqrt(12*c^2*delta_f^2/f_c^2/(4*pi^2)/0.5^2/linear_SNR/carrier_count/symbols_per_carrier1/(symbols_per_carrier1-1)/(7*symbols_per_carrier1));
% CRB_Tar3(i)=a*CRBD_3(i)+(1-a)*CRBV_3(i);
%           a=0.6; 
% CRBD_3(i)=sqrt(12*c^2/delta_f^2/(4*pi^2)/0.5^2/linear_SNR/(carrier_count*2)/symbols/((carrier_count*2)-1)/(7*(carrier_count*2)));
% CRBV_3(i)=sqrt(12*c^2*delta_f^2/f_c^2/(4*pi^2)/0.5^2/linear_SNR/carrier_count/symbols_per_carrier1/(symbols_per_carrier1-1)/(7*symbols_per_carrier1));
% CRB_Tar4(i)=a*CRBD_3(i)+(1-a)*CRBV_3(i);
% end
% figure
% x_value=[4.46*10^(-6),8.92*10^(-6),17.84*10^(-6),35.68*10^(-6),71.35*10^(-6)];
% semilogy(x_value,CRB_Tar','s--',x_value,CRB_Tar2','^--',x_value,CRB_Tar3','+--',x_value,CRB_Tar4','*--')
% axis([0 80*10^(-6) 0.02 0.15]); 
% legend('\alpha=0.3','\alpha=0.4','\alpha=0.5','\alpha=0.6')
% xlabel('Symbol duration/(s)');
% ylabel('Weighted CRLB for distance and velocity estimation');
% grid on;