#!/usr/bin/env python3
"""Build a GreenLight extension for a second, overhead LED fixture state.

The equations mirror GreenLight's toplight exchanges while keeping HPS and LED
electrical inputs separate.  Numeric LED thermal parameters are diagnostic
seeds from the GreenLight LED toplight model, not fitted AGC fixture values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def variable(unit: str, kind: str, definition: str, description: str, **extra: str) -> dict:
    result = {"unit": unit, "type": kind, "definition": definition, "description": description}
    result.update(extra)
    return result


def build_extension() -> dict:
    v = variable
    return {
        "Info": {
            "About": "Diagnostic GreenLight extension adding ELIXIA as a second overhead light source",
            "Status": "structure implemented; AGC thermal parameters uncalibrated; calibration-period use only",
            "Topology": "HPS and LED are separate overhead sources; GreenLight interlighting is not used",
            "Parameter provenance": "initial LED thermal seeds mirror Katzin-2021 LED toplights and are not AGC measurements",
        },
        "AGC overhead LED state and receiver overrides": {
            "tCovIn": v("°C", "state", "(1/capCovIn) * (hTopCovIn + lTopCovIn + rCanCovIn + rFlrCovIn + rPipeCovIn + rThScrCovIn - hCovInCovE + rBlScrCovIn + rLampCovIn + rIntLampCovIn + rLedCovIn)", "Internal cover temperature with overhead LED longwave exchange", init="16.5"),
            "tBlScr": v("°C", "state", "(1/capBlScr) * (hAirBlScr + lAirBlScr + rCanBlScr + rFlrBlScr + rPipeBlScr - hBlScrTop - rBlScrCovIn - rBlScrSky - rBlScrThScr + rLampBlScr + rIntLampBlScr + rLedBlScr)", "Blackout-screen temperature with overhead LED exchange", init="16.5"),
            "tThScr": v("°C", "state", "(1/capThScr) * (hAirThScr + lAirThScr + rCanThScr + rFlrThScr + rPipeThScr - hThScrTop - rThScrCovIn - rThScrSky + rBlScrThScr + rLampThScr + rIntLampThScr + rLedThScr)", "Thermal-screen temperature with overhead LED exchange", init="16.5"),
            "tAir": v("°C", "state", "(1/capAir) * (hCanAir + hPadAir - hAirMech + hPipeAir + hPasAir + hBlowAir + rGlobSunAir - hAirFlr - hAirThScr - hAirOut - hAirTop - hAirOutPad - lAirFog - hAirBlScr + hLampAir + rLampAir + hGroPipeAir + hIntLampAir + rIntLampAir + hLedAir + rLedAir)", "Main-air temperature with overhead LED convection and shortwave absorption", init="16.5"),
            "tCan": v("°C", "state", "(1/capCan) * (rParSunCan + rNirSunCan + rPipeCan - hCanAir - lCanAir - rCanCovIn - rCanFlr - rCanSky - rCanThScr - rCanBlScr + rParLampCan + rNirLampCan + rFirLampCan + rGroPipeCan + rParIntLampCan + rNirIntLampCan + rFirIntLampCan + rParLedCan + rNirLedCan + rFirLedCan)", "Canopy temperature with overhead LED radiation", init="20.5"),
            "tPipe": v("°C", "state", "1 / capPipe * (hBoilPipe + hIndPipe + hGeoPipe - rPipeSky - rPipeCovIn - rPipeCan - rPipeFlr - rPipeThScr - hPipeAir - rPipeBlScr + rLampPipe + rIntLampPipe + rLedPipe + hBufHotPipe)", "Rail-pipe temperature with overhead LED exchange", init="16.5"),
            "tFlr": v("°C", "state", "(1/capFlr) * (hAirFlr + rParSunFlr + rNirSunFlr + rCanFlr + rPipeFlr - hFlrSo1 - rFlrCovIn - rFlrSky - rFlrThScr - rFlrBlScr + rParLampFlr + rNirLampFlr + rFirLampFlr + rParIntLampFlr + rNirIntLampFlr + rFirIntLampFlr + rParLedFlr + rNirLedFlr + rFirLedFlr)", "Floor temperature with overhead LED radiation", init="16.5"),
            "tLed": v("°C", "state", "(1/capLed) * (qLedProcessed - rLedSky - rLedCovIn - rLedThScr - rLedBlScr - hLedAir - rParLedCan - rNirLedCan - rFirLedCan - rLedPipe - rParLedFlr - rNirLedFlr - rFirLedFlr - rLedAir - hLedCool)", "Effective temperature of the overhead ELIXIA fixture layer", init="16.5"),
        },
        "AGC observed lighting inputs and shortwave exchanges": {
            "qLampIn": v("W m**-2", "aux", "qHpsProcessed", "HPS electrical input from the explicit AGC accounting driver"),
            "rParGhLed": v("W m**-2", "aux", "ledParPhotonFlux / zetaLedPar", "Effective LED PAR above the canopy"),
            "rNirGhLed": v("W m**-2", "aux", "etaLedNir * qLedProcessed", "Effective non-PAR shortwave LED radiation"),
            "rParLedCanDown": v("W m**-2", "aux", "rParGhLed * (1-rhoCanPar) * (1-exp(-k1Par*lai))", "LED PAR directly absorbed by canopy"),
            "rParLedFlrCanUp": v("W m**-2", "aux", "rParGhLed * exp(-k1Par*lai) * rhoFlrPar * (1-rhoCanPar) * (1-exp(-k2Par*lai))", "Floor-reflected LED PAR absorbed by canopy"),
            "rParLedCan": v("W m**-2", "aux", "rParLedCanDown + rParLedFlrCanUp", "Total LED PAR absorbed by canopy"),
            "rNirLedCan": v("W m**-2", "aux", "rNirGhLed * (1-rhoCanNir) * (1-exp(-kNir*lai))", "LED non-PAR shortwave absorbed by canopy"),
            "rParLedFlr": v("W m**-2", "aux", "rParGhLed * (1-rhoFlrPar) * exp(-k1Par*lai)", "LED PAR absorbed by floor"),
            "rNirLedFlr": v("W m**-2", "aux", "rNirGhLed * (1-rhoFlrNir) * exp(-kNir*lai)", "LED non-PAR shortwave absorbed by floor"),
            "rLedAir": v("W m**-2", "aux", "rParGhLed + rNirGhLed - rParLedCan - rNirLedCan - rParLedFlr - rNirLedFlr", "LED shortwave absorbed by greenhouse air"),
            "parLedCan": v("µmol m**-2 s**-1", "aux", "ledParPhotonFlux * (rParLedCan / max(1e-9, rParGhLed))", "LED PAR photons absorbed by canopy"),
            "rParCan": v("W m**-2", "aux", "rParSunCan + rParGhLamp + rParGhIntLamp + rParGhLed", "Total PAR above and outside canopy including overhead LED"),
            "rCanLed": v("W m**-2", "aux", "rParGhLed + rNirGhLed", "Shortwave radiation above canopy from overhead LED"),
            "rCan": v("W m**-2", "aux", "rCanSun + rCanLamp + rCanIntLamp + rCanLed", "Global radiation above canopy including overhead LED"),
            "parCan": v("µmol m**-2 s**-1", "aux", "parJtoUmolSun*rParSunCan + zetaLampPar*rParLampCan + zetaIntLampPar*rParIntLampCan + parLedCan", "Total canopy-absorbed PAR photons including overhead LED"),
        },
        "AGC overhead LED longwave and convection": {
            "rLedSky": v("W m**-2", "aux", "fir(aLed,epsLedTop,epsSky,tauCovFir*tauThScrFirU*tauBlScrFirU,tLed,tSky)", "LED-to-sky FIR"),
            "rLedCovIn": v("W m**-2", "aux", "fir(aLed,epsLedTop,epsCovFir,tauThScrFirU*tauBlScrFirU,tLed,tCovIn)", "LED-to-cover FIR"),
            "rLedThScr": v("W m**-2", "aux", "fir(aLed,epsLedTop,epsThScrFir,uThScr*tauBlScrFirU,tLed,tThScr)", "LED-to-thermal-screen FIR"),
            "rLedBlScr": v("W m**-2", "aux", "fir(aLed,epsLedTop,epsBlScrFir,uBlScr,tLed,tBlScr)", "LED-to-blackout-screen FIR"),
            "rFirLedCan": v("W m**-2", "aux", "fir(aLed,epsLedBottom,epsCan,aCan,tLed,tCan)", "LED-to-canopy FIR"),
            "rLedPipe": v("W m**-2", "aux", "fir(aLed,epsLedBottom,epsPipe,tauIntLampFir*0.49*pi*lPipe*phiPipeE*exp(-kFir*lai),tLed,tPipe)", "LED-to-rail-pipe FIR"),
            "rFirLedFlr": v("W m**-2", "aux", "fir(aLed,epsLedBottom,epsFlr,tauIntLampFir*(1-0.49*pi*lPipe*phiPipeE)*exp(-kFir*lai),tLed,tFlr)", "LED-to-floor FIR"),
            "hLedAir": v("W m**-2", "aux", "sensible(cHecLedAir,tLed,tAir)", "LED convection to main air"),
            "hLedCool": v("W m**-2", "aux", "etaLedCool*qLedProcessed", "LED heat actively removed from greenhouse"),
        },
        "Uncalibrated AGC overhead LED parameter seeds": {
            "zetaLedPar": v("µmol J**-1", "const", "5.4", "Effective LED PAR photons per joule of PAR; calibration seed"),
            "etaLedNir": v("-", "const", "0.02", "Non-PAR shortwave fraction; calibration seed"),
            "capLed": v("J K**-1 m**-2", "const", "10", "Effective LED fixture heat capacity; calibration seed"),
            "cHecLedAir": v("W m**-2 K**-1", "const", "2.3", "LED-air heat exchange; calibration seed"),
            "aLed": v("m**2 m**-2", "const", "0.02", "Effective overhead LED area; calibration seed"),
            "epsLedTop": v("-", "const", "0.1", "Top-side LED emissivity; calibration seed"),
            "epsLedBottom": v("-", "const", "0.9", "Bottom-side LED emissivity; calibration seed"),
            "etaLedCool": v("-", "const", "0", "Fraction of LED input actively removed; calibration seed"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(build_extension(), indent=2) + "\n")


if __name__ == "__main__":
    main()
