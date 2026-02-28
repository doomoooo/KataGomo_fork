/opt/katago/katago benchmark -model /opt/katago/weight/b18tf.onnx -config /opt/katago/config/gtp_example.cfg -v 10000 -t 40 -fixed-batch-size 10 -override-config numNNServerThreadsPerModel=2,trtDeviceToUseThread0=0,trtDeviceToUseThread1=0

# /opt/katago/katago benchmark -model /opt/katago/weight/b18tf.onnx -config /opt/katago/config/gtp_example.cfg -v 10000 -t 40 