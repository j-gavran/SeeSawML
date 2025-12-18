# Creating ML NTuples in TNAnalysis

We need physics ROOT NTuples `SeeSawML` converter can read. For this example, we will be using [TNAnalysis](https://gitlab.cern.ch/tadej/TNAnalysis). On details of doing that, refer to TNAnalysis documentation.

To start - general idea of a preselection file is:

```python
from tnanalysis.config import ()

from tnanalysis.native_enums import ()

from ttHcc.common import SamplesML # These are samples we will work with

histEdges = linear_scale(nr_bins, from, to) # Histogram definitions. Binning quantity is auto-defined through TNAnalysis

def generate_histograms() -> list[Hist]:
    h = [
    # Hist definitions
    ]
    return add_log_plots(h)

class JobConfig(JobConfigBase):
    enableFullPreselection = False
    fillMode = Fill.DiLeptonPreselection
    triggerType = Trigger.SingleLepton
    chargeFlipMode = ChargeFlip.Egamma
    channels = [
        LeptonChannel.DiLepton,
    ]
    variations = [ # Here, you define cuts. Use one preselection variation.
        Variation(
            "preselection",
            LeptonSign.AnySign,
            ranges=[
            ],
        ),
    ]
    quantitiesML = [
    ]

    samples = SamplesML
    histograms = generate_histograms()
```

Your histogram definitions are just to check MC/Data agreements. JobConfig class is provided for a dilepton config, but is trivially expandable to 1 and 0 lepton (see comitted preselections). Every chanel works out of the box.

Quantities ML are your flat quantities; as code is designed, here exist following options:

- **Flat (event level):** enter desired event-level features into `quantitiesML` (e.g., `ptl1`); it only accepts event-level features.
- **Jagged ("set" level):** delete `quantitiesML`; you’ll get `jet_pt` and other "vectors of vectors."
- **Flat + jagged:** put desired flat/event-level features into `quantitiesML` **and** set `enableFullPreselection=True`; you will get a combination of both.

At no point you actually need to specify jagged/set level inputs such as `jet_pt`; this gets handled by `enableFullPreselection` flag.
