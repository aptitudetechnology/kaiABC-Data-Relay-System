# KaiABC Hypothesis Software Tests

This directory contains software simulations to test the mathematical hypotheses **before** deploying hardware.

## 📁 Files

### **Core Test Files**
- `test_basin_volume.py` - **MVP test** (Basin Volume Hypothesis)
- `requirements.txt` - Python dependencies
- `QUICKSTART.md` - Installation and usage guide

### **Documentation**
- `kakeya-oscillator-software-mvp.md` - Complete MVP documentation
- `README.md` - This file

## 🎯 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the MVP test (5 minutes)
python3 test_basin_volume.py
```

## 🧪 What Gets Tested

### **Current Tests (MVP)**
1. ✅ **Basin Volume** - Does σ_T → σ_ω → basin volume formula work?

### **Planned Tests (Full Suite)**
2. ⏳ Critical Coupling - Does K_c = 2σ_ω predict sync threshold?
3. ⏳ Sync Time - Does τ_sync formula predict convergence speed?
4. ⏳ Temperature Entrainment - Can coupling overcome diurnal cycles?
5. ⏳ Q10 Robustness - Does system tolerate heterogeneous Q10?
6. ⏳ Packet Loss - Can network sync with message failures?

## 📊 Expected Results

**For Q10=1.1, N=10, σ_T=5°C:**
- Predicted basin volume: **28%**
- Expected convergence rate: **26-30%**
- Error threshold: **< 15%** for acceptance

## 🚀 Next Steps

1. **Run MVP** → If passes, proceed to full suite
2. **Full Suite** → If 5/6 pass, order hardware
3. **Hardware Test** → If syncs in 16±8 days, write paper
4. **Publish** → Validate Kakeya → Kuramoto → IoT connection! 🎉

## 📖 References

See parent directory `../` for:
- `kakeya-oscillator.md` - Hardware hypothesis (30-day outdoor deployment)
- `kakeya-oscillator-software.md` - Full 6-hypothesis test plan
- `../../research/KaiABC/deep-research-prompt-claude.md` - Theoretical foundation

## 💡 Philosophy

> "If it doesn't work in simulation, it won't work in hardware."

**Cost:** $0 (software) vs. $104 (hardware)  
**Time:** 5 minutes (MVP) vs. 30 days (hardware)  
**Risk:** None vs. potential failure

**Always validate theory before building!**
