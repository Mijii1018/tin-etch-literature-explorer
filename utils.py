"""
Pure unit-conversion helpers.

不碰 Streamlit、不碰 session_state，純函式，
輸入輸出都是數字，方便之後單獨測試。
"""


def nm_min_to_A_s(rate_nm_min):
    return rate_nm_min * 10 / 60

def A_s_to_nm_min(rate_A_s):
    return rate_A_s * 60 / 10
