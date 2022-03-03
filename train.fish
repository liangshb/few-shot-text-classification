for model in cnn cnnh lstm lstm_attn gru gru_attn
  echo python main.py "sysevr_config_$model.yaml"
  python main.py "sysevr_config_$model.yaml"
end

