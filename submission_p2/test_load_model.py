import pytest

from .load_model import load_model

models = [
    "xx_chu_sigtyp_trf",
    "el_cop_sigtyp_trf",
    "xx_fro_sigtyp_trf",
    "xx_got_sigtyp_trf",
    "xx_grc_sigtyp_trf",
    "he_hbo_sigtyp_trf",
    "xx_isl_sigtyp_trf",
    "xx_lat_sigtyp_trf",
    "xx_latm_sigtyp_trf",
    "zh_lzh_sigtyp_trf",
    "xx_ohu_sigtyp_trf",
    "xx_orv_sigtyp_trf",
    "xx_san_sigtyp_trf",
]


@pytest.mark.parametrize("model", models)
def test_get_embedding(model):
    test_word = "hocum"  # should still return a value even if not in a language's vocab
    trf_model = load_model(model)
    embedding = trf_model[test_word]
    assert embedding.shape == (768,)
