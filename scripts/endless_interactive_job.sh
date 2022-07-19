while :
do
qrsh -g gcc50556 -l rt_AG.small=1 -l h_rt=12:00:00 bash /home/acc12675ut/D1/NARF-GAN-dev/scripts/train_SSO_abci.sh
done