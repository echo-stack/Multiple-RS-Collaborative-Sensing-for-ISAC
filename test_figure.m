
%  % 一、不同参考信号测距，1、PRS 2、DMRS 3、PRS和DMRS
% f_c2=24*10^9;
% ma1=1;
% symbols_per_carrier1=140;
% carrier_count1=128; 
% delta_f1=120*10^3;
% symbols=symbols_per_carrier1*(4/14);
% carrier_count=carrier_count1;
% delta_f=delta_f1;
% f_c=f_c2;
% c = 3*10^8;
% for i=1:26
%          SNR1=-16+1*i;
% RMSE1(i)=PRS_r(SNR1,f_c2,ma1,symbols_per_carrier1,carrier_count1,delta_f1);
% RMSE2(i)=DMRS_r(SNR1,f_c2,ma1,symbols_per_carrier1,carrier_count1,delta_f1);
% RMSE3(i)=joint_test_r(SNR1,f_c2,symbols_per_carrier1,carrier_count1,delta_f1);
% linear_SNR=10^(SNR1/10);  %转换对数信噪比为线性幅度值
% CRBD_1(i)=sqrt(12*c^2/delta_f^2/(4*pi^2)/0.5^2/linear_SNR/symbols_per_carrier1/(carrier_count*2)/(carrier_count-1)/(7*carrier_count+1));
% CRBD_2(i)=sqrt(12*c^2/delta_f^2/(4*pi^2)/0.5^2/linear_SNR/2/symbols/(carrier_count*2)/(carrier_count-1)/(7*carrier_count+1));
% CRBD_3(i)=sqrt(12*c^2/delta_f^2/(4*pi^2)/0.5^2/linear_SNR/(carrier_count*2)/symbols/((carrier_count*2)-1)/(7*(carrier_count*2)));
% end
% figure
% semilogy(linspace(-15,10,26),RMSE1','-m',linspace(-15,10,26),RMSE2','-b',linspace(-15,10,26),RMSE3','-g',linspace(-15,10,26),CRBD_1',':m',linspace(-19,10,26),CRBD_2',':b',linspace(-15,10,26),CRBD_3',':g')
% axis([-15 10 0.005 250]); 
% legend('PRS','DMRS','Combine PRS and DMRS','Root CRLB based PRS','Root CRLB based DMRS','Root CRLB based PRS and DMRS ')
% xlabel('SNR/（dB）');
% ylabel('RMSE of distance estimation/(m)');                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

% % %  不同子载波间隔
% f_c2=24*10^9;
% ma1=1;
% symbols_per_carrier1=140;
% carrier_count1=128;
% for i=1:21
%          SNR1=-16+1*i;
% 
% RMSE2(i)=joint_test_r(SNR1,f_c2,symbols_per_carrier1,carrier_count1,30*10^3);
% RMSE3(i)=joint_test_r(SNR1,f_c2,symbols_per_carrier1,carrier_count1,60*10^3);
% RMSE4(i)=joint_test_r(SNR1,f_c2,symbols_per_carrier1,carrier_count1,120*10^3);
% RMSE5(i)=joint_test_r(SNR1,f_c2,symbols_per_carrier1,carrier_count1,240*10^3);
% 
% 
% end
% figure
% semilogy(linspace(-15,5,21),RMSE2','s--',linspace(-15,5,21),RMSE3','^--',linspace(-15,5,21),RMSE4','+--',linspace(-15,5,21),RMSE5','*--')
% axis([-15 5 0.2 280]); 
% legend('30kHz','60kHz','120kHz','240kHz')
% xlabel('SNR/（dB）');
% ylabel('RMSE of distance estimation/(m)');


% % 不同参考信号，1、PRS 2、DMRS 3、PRS和DMRS,CSI-RS.测速
% f_c2=24*10^9;
% ma1=1;
% symbols_per_carrier1=140;
% carrier_count1=128; 
% delta_f1=120*10^3;
% symbols=symbols_per_carrier1*(4/14);
% carrier_count=carrier_count1;
% delta_f=delta_f1;
% f_c=f_c2;
% c = 3*10^8;
% for i=1:21
%          SNR1=-11+1*i;
% RMSE1(i)=PRS_v(SNR1,f_c2,ma1,symbols_per_carrier1,carrier_count1,delta_f1);
% RMSE2(i)=DMRS_v(SNR1,f_c2,ma1,symbols_per_carrier1,carrier_count1,delta_f1);
% RMSE3(i)=joint_test_v(SNR1,f_c2,symbols_per_carrier1,carrier_count1,delta_f1);
% linear_SNR=10^(SNR1/10);  %转换对数信噪比为线性幅度值
% CRBV_1(i)=sqrt(12*c^2*delta_f^2/f_c^2/(4*pi^2)/0.5^2/linear_SNR/symbols_per_carrier1/(carrier_count*2)/(symbols_per_carrier1/2-1)/(7*symbols_per_carrier1/2+1));
% CRBV_2(i)=sqrt(12*c^2*delta_f^2/f_c^2/(4*pi^2)/0.5^2/linear_SNR/(14/4)/carrier_count/symbols_per_carrier1/(symbols-1)/(7*symbols+1));
% CRBV_3(i)=sqrt(12*c^2*delta_f^2/f_c^2/(4*pi^2)/0.5^2/linear_SNR/carrier_count/symbols_per_carrier1/(symbols_per_carrier1-1)/(7*symbols_per_carrier));
% end
% figure
% semilogy(linspace(-10,10,21),RMSE1','-m',linspace(-10,10,21),RMSE2','-b',linspace(-10,10,21),RMSE3','-g',linspace(-10,10,21),CRBV_1',':m',linspace(-10,10,21),CRBV_2',':b',linspace(-10,10,21),CRBV_3',':g')
% axis([-10 10 0.01 150]); 
% legend('PRS','DMRS','Combine PRS,DMRS and CSI-RS','Root CRLB based PRS','Root CRLB based DMRS','Root CRLB based PRS,DMRS and CSI-RS')
% xlabel('SNR/（dB）');
% ylabel('RMSE of velocity estimation/(m/s)');

 % % 中心频率
% f_c=[5.9*10^9 24*10^9];
% f_c1=5.9*10^9;
% f_c2=24*10^9;
% ma1=1;
% symbols_per_carrier1=140;
% carrier_count1=128;
% delta_f1=120*10^3;
% for i=1:21
%          SNR1=-11+1*i;
% RMSE1(i)=joint_test_v(SNR1,f_c1,symbols_per_carrier1,carrier_count1,delta_f1);
% RMSE2(i)=joint_test_v(SNR1,f_c2,symbols_per_carrier1,carrier_count1,delta_f1);
% end
% figure
% semilogy(linspace(-10,10,21),RMSE1','--o',linspace(-10,10,21),RMSE2','d-')
% axis([-10 10 1 150]); 
% legend('5.9GHz','24GHz')
% xlabel('SNR/（dB）');
% ylabel('RMSE of velocity/(m/s)');

%  不同信号长度
% f_c2=24*10^9;
% ma1=1;
% symbols_per_carrier1=140;
% carrier_count1=128;
% for i=1:21
%          SNR1=-16+1*i;

% RMSE2(i)=joint_test_v(SNR1,f_c2,symbols_per_carrier1,carrier_count1,30*10^3);
% RMSE3(i)=joint_test_v(SNR1,f_c2,symbols_per_carrier1,carrier_count1,60*10^3);
% RMSE4(i)=joint_test_v(SNR1,f_c2,symbols_per_carrier1,carrier_count1,120*10^3);
% RMSE5(i)=joint_test_v(SNR1,f_c2,symbols_per_carrier1,carrier_count1,240*10^3);
% end
% figure
% semilogy(linspace(-15,5,21),RMSE2','s--',linspace(-15,5,21),RMSE3','^--',linspace(-15,5,21),RMSE4','+--',linspace(-15,5,21),RMSE5','*--')
% axis([-15 5 0.2 200]); 
% legend('35.68us','17.84us','8.92us','4.46us')
% xlabel('SNR/（dB）');
% ylabel('RMSE of distance estimation/(m/s)');

%  测距和测速的不同信号长度
% f_c2=24*10^9;
% ma1=1;
% symbols_per_carrier1=140;
% carrier_count1=128;
% for i=1:3
%          delta_f1=30*10^3*(2^i);
% RMSE2(i)=joint_test_v(-12,f_c2,symbols_per_carrier1,carrier_count1,delta_f1);
% RMSE3(i)=joint_test_v(-8,f_c2,symbols_per_carrier1,carrier_count1,delta_f1);
% RMSE4(i)=joint_test_v(-4,f_c2,symbols_per_carrier1,carrier_count1,delta_f1);
% RMSE5(i)=joint_test_v(0,f_c2,symbols_per_carrier1,carrier_count1,delta_f1);
% 
% RMSE2_R(i)=joint_test_r(-12,f_c2,symbols_per_carrier1,carrier_count1,delta_f1);
% RMSE3_R(i)=joint_test_r(-8,f_c2,symbols_per_carrier1,carrier_count1,delta_f1);
% RMSE4_R(i)=joint_test_r(-4,f_c2,symbols_per_carrier1,carrier_count1,delta_f1);
% RMSE5_R(i)=joint_test_r(0,f_c2,symbols_per_carrier1,carrier_count1,delta_f1);
% end
% figure
% loglog(RMSE2,RMSE2_R','s--',RMSE3,RMSE3_R','^--',RMSE4,RMSE4_R','+--',RMSE5,RMSE5_R','*--')
% axis([0.1 290 0.1 120]); 
% legend('35.68us','17.84us','8.92us','4.46us')
% xlabel('RMSE of velocity estimation/(m/s)');
% ylabel('RMSE of distance estimation/(m)');

%测距测速CRLB关于符号长度的权衡考虑
f_c2=24*10^9;
ma1=1;
symbols_per_carrier1=140;
carrier_count1=128; 
symbols=symbols_per_carrier1*(4/14);
carrier_count=carrier_count1;
f_c=f_c2;
c = 3*10^8;
SNR=0;
linear_SNR=10^(SNR/10);
for i=1:5
         delta_f1=15*10^3*(2^(5-i));
         delta_f=delta_f1;
         
         a=0.3;
CRBD_3(i)=sqrt(12*c^2/delta_f^2/(4*pi^2)/0.5^2/linear_SNR/(carrier_count*2)/symbols/((carrier_count*2)-1)/(7*(carrier_count*2)));
CRBV_3(i)=sqrt(12*c^2*delta_f^2/f_c^2/(4*pi^2)/0.5^2/linear_SNR/carrier_count/symbols_per_carrier1/(symbols_per_carrier1-1)/(7*symbols_per_carrier1));
CRB_Tar(i)=a*CRBD_3(i)+(1-a)*CRBV_3(i);
        a=0.4;
CRBD_3(i)=sqrt(12*c^2/delta_f^2/(4*pi^2)/0.5^2/linear_SNR/(carrier_count*2)/symbols/((carrier_count*2)-1)/(7*(carrier_count*2)));
CRBV_3(i)=sqrt(12*c^2*delta_f^2/f_c^2/(4*pi^2)/0.5^2/linear_SNR/carrier_count/symbols_per_carrier1/(symbols_per_carrier1-1)/(7*symbols_per_carrier1));
CRB_Tar2(i)=a*CRBD_3(i)+(1-a)*CRBV_3(i);
        a=0.5;  
CRBD_3(i)=sqrt(12*c^2/delta_f^2/(4*pi^2)/0.5^2/linear_SNR/(carrier_count*2)/symbols/((carrier_count*2)-1)/(7*(carrier_count*2)));
CRBV_3(i)=sqrt(12*c^2*delta_f^2/f_c^2/(4*pi^2)/0.5^2/linear_SNR/carrier_count/symbols_per_carrier1/(symbols_per_carrier1-1)/(7*symbols_per_carrier1));
CRB_Tar3(i)=a*CRBD_3(i)+(1-a)*CRBV_3(i);
          a=0.6; 
CRBD_3(i)=sqrt(12*c^2/delta_f^2/(4*pi^2)/0.5^2/linear_SNR/(carrier_count*2)/symbols/((carrier_count*2)-1)/(7*(carrier_count*2)));
CRBV_3(i)=sqrt(12*c^2*delta_f^2/f_c^2/(4*pi^2)/0.5^2/linear_SNR/carrier_count/symbols_per_carrier1/(symbols_per_carrier1-1)/(7*symbols_per_carrier1));
CRB_Tar4(i)=a*CRBD_3(i)+(1-a)*CRBV_3(i);
end
figure
x_value=[4.46*10^(-6),8.92*10^(-6),17.84*10^(-6),35.68*10^(-6),71.35*10^(-6)];
semilogy(x_value,CRB_Tar','s--',x_value,CRB_Tar2','^--',x_value,CRB_Tar3','+--',x_value,CRB_Tar4','*--')
axis([0 80*10^(-6) 0.02 0.15]); 
legend('\alpha=0.3','\alpha=0.4','\alpha=0.5','\alpha=0.6')
xlabel('Symbol duration/(s)');
ylabel('Weighted CRLB for distance and velocity estimation');
grid on;