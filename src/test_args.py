from dataclasses import dataclass


# Tor Dataset
@dataclass
class TorDTArgs:
    dis = "dt"
    trained_dis_path = "saved_models/tor_dt_dis.pkl"
    trained_amoeba_path = "saved_models/tor_dt_amoeba.zip"
    encoder_path = "saved_models/encoder.pth"
    data_config = "config/tor_data.json"
    enc_dim = 256
    layer_num = 2
    max_episode_length = 60
    MAX_UNIT = 1448
    MAX_DELAY = 1
    adv_pkt_clip = 1
    adv_iat_clip = 0.001
    test_iat_scale = 1
    test_num = 2000
    action_mode = "direction_limited"


@dataclass
class TorRFArgs:
    dis = "rf"
    trained_dis_path = "saved_models/tor_rf_dis.pkl"
    trained_amoeba_path = "saved_models/tor_rf_amoeba.zip"
    encoder_path = "saved_models/encoder.pth"
    data_config = "config/tor_data.json"
    enc_dim = 256
    layer_num = 2
    max_episode_length = 60
    MAX_UNIT = 1448
    MAX_DELAY = 1
    adv_pkt_clip = 1
    adv_iat_clip = 0.001
    test_iat_scale = 1
    test_num = 2000
    action_mode = "direction_limited"


@dataclass
class TorCUMULArgs:
    dis = "cumul"
    trained_dis_path = "saved_models/tor_cumul_dis.pkl"
    trained_amoeba_path = "saved_models/tor_cumul_amoeba.zip"
    encoder_path = "saved_models/encoder.pth"
    data_config = "config/tor_data.json"
    enc_dim = 256
    layer_num = 2
    max_episode_length = 60
    MAX_UNIT = 1448
    MAX_DELAY = 1
    adv_pkt_clip = 1
    adv_iat_clip = 0.001
    test_iat_scale = 1
    test_num = 2000
    action_mode = "direction_limited"


@dataclass
class TorDFArgs:
    dis = "df"
    trained_dis_path = "saved_models/tor_df_dis.pth"
    trained_amoeba_path = "saved_models/tor_df_amoeba.zip"
    encoder_path = "saved_models/encoder.pth"
    data_config = "config/tor_data.json"
    enc_dim = 256
    layer_num = 2
    max_episode_length = 60
    MAX_UNIT = 1448
    MAX_DELAY = 1
    adv_pkt_clip = 1
    adv_iat_clip = 0.001
    test_iat_scale = 1
    test_num = 2000
    action_mode = "direction_limited"


@dataclass
class TorSDAEArgs:
    dis = "sdae"
    trained_dis_path = "saved_models/tor_sdae_dis.pth"
    trained_amoeba_path = "saved_models/tor_sdae_amoeba.zip"
    encoder_path = "saved_models/encoder.pth"
    data_config = "config/tor_data.json"
    enc_dim = 256
    layer_num = 2
    max_episode_length = 60
    MAX_UNIT = 1448
    MAX_DELAY = 1
    adv_pkt_clip = 1
    adv_iat_clip = 0.001
    test_iat_scale = 1
    test_num = 2000
    action_mode = "direction_limited"


@dataclass
class TorLSTMArgs:
    dis = "lstm"
    trained_dis_path = "saved_models/tor_lstm_dis.pth"
    trained_amoeba_path = "saved_models/tor_lstm_amoeba.zip"
    encoder_path = "saved_models/encoder.pth"
    data_config = "config/tor_data.json"
    enc_dim = 256
    layer_num = 2
    max_episode_length = 60
    MAX_UNIT = 1448
    MAX_DELAY = 1
    adv_pkt_clip = 1
    adv_iat_clip = 0.001
    test_iat_scale = 1
    test_num = 2000
    action_mode = "direction_limited"


# V2ray Dataset
@dataclass
class V2rayDTArgs:
    dis = "dt"
    trained_dis_path = "saved_models/v2ray_dt_dis.pkl"
    trained_amoeba_path = "saved_models/v2ray_dt_amoeba.zip"
    encoder_path = "saved_models/encoder.pth"
    data_config = "config/v2ray_data.json"
    enc_dim = 256
    layer_num = 2
    max_episode_length = 60
    MAX_UNIT = 16500
    MAX_DELAY = 1
    adv_pkt_clip = 1
    adv_iat_clip = 0.001
    test_iat_scale = 1
    test_num = 2000
    action_mode = "direction_limited"


@dataclass
class V2rayRFArgs:
    dis = "rf"
    trained_dis_path = "saved_models/v2ray_rf_dis.pkl"
    trained_amoeba_path = "saved_models/v2ray_rf_amoeba.zip"
    encoder_path = "saved_models/encoder.pth"
    data_config = "config/v2ray_data.json"
    enc_dim = 256
    layer_num = 2
    max_episode_length = 60
    MAX_UNIT = 16500
    MAX_DELAY = 1
    adv_pkt_clip = 1
    adv_iat_clip = 0.001
    test_iat_scale = 1
    test_num = 2000
    action_mode = "direction_limited"


@dataclass
class V2rayCUMULArgs:
    dis = "cumul"
    trained_dis_path = "saved_models/v2ray_cumul_dis.pkl"
    trained_amoeba_path = "saved_models/v2ray_cumul_amoeba.zip"
    encoder_path = "saved_models/encoder.pth"
    data_config = "config/v2ray_data.json"
    enc_dim = 256
    layer_num = 2
    max_episode_length = 60
    MAX_UNIT = 16500
    MAX_DELAY = 1
    adv_pkt_clip = 1
    adv_iat_clip = 0.001
    test_iat_scale = 1
    test_num = 2000
    action_mode = "direction_limited"


@dataclass
class V2rayDFArgs:
    dis = "df"
    trained_dis_path = "saved_models/v2ray_df_dis.pth"
    trained_amoeba_path = "saved_models/v2ray_df_amoeba.zip"
    encoder_path = "saved_models/encoder.pth"
    data_config = "config/v2ray_data.json"
    enc_dim = 256
    layer_num = 2
    max_episode_length = 60
    MAX_UNIT = 16500
    MAX_DELAY = 1
    adv_pkt_clip = 1
    adv_iat_clip = 0.001
    test_iat_scale = 1
    test_num = 2000
    action_mode = "direction_limited"


@dataclass
class V2raySDAEArgs:
    dis = "sdae"
    trained_dis_path = "saved_models/v2ray_sdae_dis.pth"
    trained_amoeba_path = "saved_models/v2ray_sdae_amoeba.zip"
    encoder_path = "saved_models/encoder.pth"
    data_config = "config/v2ray_data.json"
    enc_dim = 256
    layer_num = 2
    max_episode_length = 60
    MAX_UNIT = 16500
    MAX_DELAY = 1
    adv_pkt_clip = 1
    adv_iat_clip = 0.001
    test_iat_scale = 1
    test_num = 2000
    action_mode = "direction_limited"


@dataclass
class V2rayLSTMArgs:
    dis = "lstm"
    trained_dis_path = "saved_models/v2ray_lstm_dis.pth"
    trained_amoeba_path = "saved_models/v2ray_lstm_amoeba.zip"
    encoder_path = "saved_models/encoder.pth"
    data_config = "config/v2ray_data.json"
    enc_dim = 256
    layer_num = 2
    max_episode_length = 60
    MAX_UNIT = 16500
    MAX_DELAY = 1
    adv_pkt_clip = 1
    adv_iat_clip = 0.001
    test_iat_scale = 1
    test_num = 2000
    action_mode = "direction_limited"


test_args_dict = {
    "tor": {
        "dt": TorDTArgs(),
        "rf": TorRFArgs(),
        "cumul": TorCUMULArgs(),
        "df": TorDFArgs(),
        "sdae": TorSDAEArgs(),
        "lstm": TorLSTMArgs()
    },
    "v2ray": {
        "dt": V2rayDTArgs(),
        "rf": V2rayRFArgs(),
        "cumul": V2rayCUMULArgs(),
        "df": V2rayDFArgs(),
        "sdae": V2raySDAEArgs(),
        "lstm": V2rayLSTMArgs()
    },
}
