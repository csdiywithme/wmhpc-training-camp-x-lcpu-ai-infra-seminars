import pytest
import torch

pytest.importorskip("tilelang", reason="需要安装 tilelang（uv sync --extra tilelang）")

if not torch.cuda.is_available():
    pytest.skip("TileLang 题需要 GPU，在集群上运行", allow_module_level=True)

from kernels.tilelang_softmax import softmax


def test_basic():
    torch.manual_seed(0)
    x = torch.randn(33, 127, device="cuda")
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=-1),
                               atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("n", [1000, 1025, 4095, 4096])
def test_wide_rows(n):
    torch.manual_seed(1)
    x = torch.randn(8, n, device="cuda")
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=-1),
                               atol=1e-5, rtol=1e-5)


def test_negative_ragged_rows():
    # 若越界列错误地补 0，行最大值会被抬高，exp(-1000) 下溢后产生 0/0。
    x = torch.full((3, 33), -1000.0, device="cuda")
    got = softmax(x)
    assert torch.isfinite(got).all(), "padding 不应影响真实行的最大值与分母"
    torch.testing.assert_close(got, torch.softmax(x, dim=-1),
                               atol=1e-5, rtol=1e-5)


def test_numerical_stability():
    # 数值巨大的一行。不先减最大值的实现，exp 会溢出成 inf/nan。
    torch.manual_seed(2)
    x = torch.randn(4, 256, device="cuda") * 1000.0
    got = softmax(x)
    assert torch.isfinite(got).all(), "出现 inf/nan——先减去行内最大值再做 exp"
    torch.testing.assert_close(got, torch.softmax(x, dim=-1),
                               atol=1e-5, rtol=1e-5)


def test_single_element_rows():
    x = torch.randn(5, 1, device="cuda")
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=-1),
                               atol=1e-5, rtol=1e-5)
