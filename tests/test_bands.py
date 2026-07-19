from kanfood.bands import assign_band


def test_known_ftir_bands():
    assert "ester" in assign_band(1743)["assignment"].lower()      # C=O triglyceride
    assert "c-h" in assign_band(2920)["assignment"].lower()        # CH2 asym
    assert assign_band(500)["assignment"] == "Unassigned"
