from dataclasses import dataclass


@dataclass(frozen=True)
class MCSample:
    name: str
    dsids: list[int]


class DSIDStore:
    def __init__(self, mc_samples: list[str]) -> None:
        self._dsid_to_sample: dict[int, str] = {}
        self._sample_to_dsids: dict[str, list[int]] = {}

        for sample_name in mc_samples:
            mc_sample = switch_on_sample(sample_name)
            dsids = mc_sample.dsids

            self._sample_to_dsids[sample_name] = dsids
            for dsid in dsids:
                if dsid in self._dsid_to_sample:
                    raise ValueError(f"Duplicate DSID {dsid} found in process {sample_name}")
                self._dsid_to_sample[dsid] = sample_name

    @property
    def all_dsids(self) -> list[int]:
        return list(self._dsid_to_sample.keys())

    @property
    def all_samples(self) -> list[str]:
        return list(self._sample_to_dsids.keys())

    def get_dsids(self, sample_name: str) -> list[int]:
        if sample_name not in self._sample_to_dsids:
            raise ValueError(f"Sample name {sample_name} not found in DSIDStore")
        return self._sample_to_dsids[sample_name]

    def get_sample(self, dsid: int) -> str:
        if dsid not in self._dsid_to_sample:
            raise ValueError(f"DSID {dsid} not found in DSIDStore")
        return self._dsid_to_sample[dsid]

    def get(self, item: int | str) -> str | list[int]:
        if isinstance(item, int):
            return self.get_sample(item)
        elif isinstance(item, str):
            return self.get_dsids(item)
        else:
            raise ValueError(f"Invalid type for DSIDStore get: {type(item)}")

    def __getitem__(self, item: int | str) -> str | list[int]:
        return self.get(item)

    def __contains__(self, item: int | str) -> bool:
        if isinstance(item, int):
            return item in self._dsid_to_sample
        elif isinstance(item, str):
            return item in self._sample_to_dsids
        else:
            raise ValueError(f"Invalid type for DSIDStore __contains__: {type(item)}")


def switch_on_sample(name: str) -> MCSample:
    match name:
        case "VH":
            return MCSample(
                name="VH",
                dsids=[
                    345053,  # Powheg+Pythia8 WmH, W->lv, H->bb
                    345054,  # Powheg+Pythia8 WpH, W->lv, H->bb
                    345055,  # Powheg+Pythia8 ZH, Z->ll, H->bb
                    345056,  # Powheg+Pythia8 ZH, Z->vv, H->bb
                    345057,  # Powheg+Pythia8 ggZH, Z->ll, H->bb
                    345109,  # Powheg+Pythia8 WmH, W->lv, H->cc
                    345110,  # Powheg+Pythia8 WpH, W->lv, H->cc
                    345111,  # Powheg+Pythia8 ZH, Z->ll, H->cc
                    345112,  # Powheg+Pythia8 ZH, Z->vv, H->cc
                    345113,  # Powheg+Pythia8 ggZH, Z->ll, H->cc
                ],
            )
        case "W+jets":
            return MCSample(
                name="W+jets",
                dsids=[
                    700341,  # Sherpa 2.2.11 Wmunu MAXHTPTV (BFilter)
                    700342,  # Sherpa 2.2.11 Wmunu MAXHTPTV (CFilterBVeto)
                    700343,  # Sherpa 2.2.11 Wmunu MAXHTPTV (CVetoBVeto)
                    700338,  # Sherpa 2.2.11 Wenu MAXHTPTV (BFilter)
                    700339,  # Sherpa 2.2.11 Wenu MAXHTPTV (CFilterBVeto)
                    700340,  # Sherpa 2.2.11 Wenu MAXHTPTV (CVetoBVeto)
                    700344,  # Sherpa 2.2.11 Wtaunu L MAXHTPTV (BFilter)
                    700345,  # Sherpa 2.2.11 Wtaunu L MAXHTPTV (CFilterBVeto)
                    700346,  # Sherpa 2.2.11 Wtaunu L MAXHTPTV (CVetoBVeto)
                    700347,  # Sherpa 2.2.11 Wtaunu H MAXHTPTV (BFilter)
                    700348,  # Sherpa 2.2.11 Wtaunu H MAXHTPTV (CFilterBVeto)
                    700349,  # Sherpa 2.2.11 Wtaunu H MAXHTPTV (CVetoBVeto)
                    700362,  # Sherpa 2.2.11 Wenu2jets VBF
                    700363,  # Sherpa 2.2.11 Wmunu2jets VBF
                    700364,  # Sherpa 2.2.11 Wtaunu2jets VBF
                ],
            )
        case "W+jets_FxFx":
            return MCSample(
                name="W+jets_FxFx",
                dsids=[
                    508979,  # MGPy8EG_Wenu_FxFx_3jets_HT2bias_BFilter
                    508980,  # MGPy8EG_Wenu_FxFx_3jets_HT2bias_CFilterBVeto
                    508981,  # MGPy8EG_Wenu_FxFx_3jets_HT2bias_CVetoBVeto
                    508982,  # MGPy8EG_Wmunu_FxFx_3jets_HT2bias_BFilter
                    508983,  # MGPy8EG_Wmunu_FxFx_3jets_HT2bias_CFilterBVeto
                    508984,  # MGPy8EG_Wmunu_FxFx_3jets_HT2bias_CVetoBVeto
                    509751,  # MGPy8EG_FxFx_Wtaunu_L_3jets_HT2bias_BFilter
                    509752,  # MGPy8EG_FxFx_Wtaunu_L_3jets_HT2bias_CFilterBVeto
                    509753,  # MGPy8EG_FxFx_Wtaunu_L_3jets_HT2bias_CVetoBVeto
                    509754,  # MGPy8EG_FxFx_Wtaunu_H_3jets_HT2bias_BFilter
                    509755,  # MGPy8EG_FxFx_Wtaunu_H_3jets_HT2bias_CFilterBVeto
                    509756,  # MGPy8EG_FxFx_Wtaunu_H_3jets_HT2bias_CVetoBVeto
                ],
            )
        case "W+jets_allhad":
            return MCSample(
                name="W+jets_allhad",
                dsids=[
                    700843,  # Sh_2214_Wqq_ptW_200_ECMS
                ],
            )
        case "W_PowhegPythia":
            return MCSample(
                name="W_PowhegPythia",
                dsids=[
                    361100,  # PowhegPythia8EvtGen_AZNLOCTEQ6L1_Wplusenu
                    361101,  # PowhegPythia8EvtGen_AZNLOCTEQ6L1_Wplusmunu
                    361102,  # PowhegPythia8EvtGen_AZNLOCTEQ6L1_Wplustaunu
                    361103,  # PowhegPythia8EvtGen_AZNLOCTEQ6L1_Wminusenu
                    361104,  # PowhegPythia8EvtGen_AZNLOCTEQ6L1_Wminusmunu
                    361105,  # PowhegPythia8EvtGen_AZNLOCTEQ6L1_Wminustaunu
                ],
            )
        case "Z+jets":
            return MCSample(
                name="Z+jets",
                dsids=[
                    700323,  # Sherpa 2.2.11 Zmumu MAXHTPTV (BFilter)
                    700324,  # Sherpa 2.2.11 Zmumu MAXHTPTV (CFilterBVeto)
                    700325,  # Sherpa 2.2.11 Zmumu MAXHTPTV (CVetoBVeto)
                    700320,  # Sherpa 2.2.11 Zee MAXHTPTV (BFilter)
                    700321,  # Sherpa 2.2.11 Zee MAXHTPTV (CFilterBVeto)
                    700322,  # Sherpa 2.2.11 Zee MAXHTPTV (CVetoBVeto)
                    700792,  # Sherpa 2.2.14 Ztautau MAXHTPTV (BFilter)
                    700793,  # Sherpa 2.2.14 Ztautau MAXHTPTV (CFilterBVeto)
                    700794,  # Sherpa 2.2.14 Ztautau MAXHTPTV (CVetoBVeto)
                    700358,  # Sherpa 2.2.11 Zee2jets VBF
                    700359,  # Sherpa 2.2.11 Zmm2jets VBF
                    700360,  # Sherpa 2.2.11 Ztt2jets VBF
                    700361,  # Sherpa 2.2.11 Znunu2jets VBF
                    # 700901,  # Ztt_maxHTpTV2_Mll10_40_BFilter
                    # 700902,  # Ztt_maxHTpTV2_Mll10_40_CFilterBVeto
                    # 700903,  # Ztt_maxHTpTV2_Mll10_40_CVetoBVeto
                ],
            )
        case "Z+jets_FxFx":
            return MCSample(
                name="Z+jets_FxFx",
                dsids=[
                    506193,  # MGPy8EG_Zee_FxFx_3jets_HT2bias_BFilter
                    506194,  # MGPy8EG_Zee_FxFx_3jets_HT2bias_CFilterBVeto
                    506195,  # MGPy8EG_Zee_FxFx_3jets_HT2bias_CVetoBVeto
                    506196,  # MGPy8EG_Zmumu_FxFx_3jets_HT2bias_BFilter
                    506197,  # MGPy8EG_Zmumu_FxFx_3jets_HT2bias_CFilterBVeto
                    506198,  # MGPy8EG_Zmumu_FxFx_3jets_HT2bias_CVetoBVeto
                    512198,  # MGPy8EG_FxFx_Ztautau_3jets_HT2bias_BFilter
                    512199,  # MGPy8EG_FxFx_Ztautau_3jets_HT2bias_CFilterBVeto
                    512200,  # MGPy8EG_FxFx_Ztautau_3jets_HT2bias_CVetoBVeto
                ],
            )
        case "Z+jets_allhad":
            return MCSample(
                name="Z+jets_allhad",
                dsids=[
                    700849,  # Sh_2214_Zqq_ptZ_200_ECMS
                ],
            )
        case "Z+jets_bb":
            return MCSample(
                name="Z+jets_bb",
                dsids=[
                    700855,  # Sh_2214_Zbb_ptZ_200_ECMS
                ],
            )
        case "Z+jets_light":
            return MCSample(
                name="Z+jets_light",
                dsids=[
                    700467,  # Sherpa 2.2.11 Zee Mll 10-40 MAXHTPTV (BFilter)
                    700468,  # Sherpa 2.2.11 Zee Mll 10-40 MAXHTPTV (CFilterBVeto)
                    700469,  # Sherpa 2.2.11 Zee Mll 10-40 MAXHTPTV (CVetoBVeto)
                    700470,  # Sherpa 2.2.11 Zmumu Mll 10-40 MAXHTPTV (BFilter)
                    700471,  # Sherpa 2.2.11 Zmumu Mll 10-40 MAXHTPTV (CFilterBVeto)
                    700472,  # Sherpa 2.2.11 Zmumu Mll 10-40 MAXHTPTV (CVetoBVeto)
                    700901,  # Sherpa 2.2.14 Ztautau Mll 10-40 MAXHTPTV (BFilter)
                    700902,  # Sherpa 2.2.14 Ztautau Mll 10-40 MAXHTPTV (CFilterBVeto)
                    700903,  # Sherpa 2.2.14 Ztautau Mll 10-40 MAXHTPTV (CVetoBVeto)
                ],
            )
        case "Z_PowhegPythia":
            return MCSample(
                name="Z_PowhegPythia",
                dsids=[
                    361106,  # PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zee
                    361107,  # PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zmumu
                    361108,  # PowhegPythia8EvtGen_AZNLOCTEQ6L1_Ztautau
                ],
            )
        case "diboson":
            return MCSample(
                name="diboson",
                dsids=[
                    701040,  # Sherpa 2.2.16 llll
                    701045,  # Sherpa 2.2.16 lllv
                    701050,  # Sherpa 2.2.16 llvv os
                    701055,  # Sherpa 2.2.16 llvv ss
                    701060,  # Sherpa 2.2.16 lvvv
                    701065,  # Sherpa 2.2.16 vvvv
                    701085,  # Sherpa 2.2.14 ZqqZll
                    701090,  # Sherpa 2.2.14 ZbbZll
                    701095,  # Sherpa 2.2.14 ZqqZvv
                    701100,  # Sherpa 2.2.14 ZbbZvv
                    701105,  # Sherpa 2.2.14 WqqZll
                    701110,  # Sherpa 2.2.14 WqqZvv
                    701115,  # Sherpa 2.2.14 WlvZqq
                    701120,  # Sherpa 2.2.14 WlvZbb
                    701125,  # Sherpa 2.2.14 WlvWqq
                    701000,  # Sherpa 2.2.14 lllljj EW
                    701005,  # Sherpa 2.2.14 lllvjj EW
                    701010,  # Sherpa 2.2.14 llvvjj os EW
                    701015,  # Sherpa 2.2.14 llvvjj ss EW
                    701020,  # Sherpa 2.2.14 lllljj QCD/EW
                    701025,  # Sherpa 2.2.14 lllvjj QCD/EW
                    701030,  # Sherpa 2.2.14 llvvjj os QCD/EW
                    701035,  # Sherpa 2.2.14 llvvjj ss QCD/EW
                ],
            )
        case "diboson_opendata":
            return MCSample(
                name="diboson_opendata",
                dsids=[
                    700587,  # Sh_2212_lllljj
                    700593,  # Sh_2212_llvvjj_os_Int
                    700710,  # Sh_2212_llgammajj
                    700604,  # Sh_2212_lvvv
                    700605,  # Sh_2212_vvvv
                    700594,  # Sh_2212_llvvjj_ss_Int
                    700600,  # Sh_2212_llll
                    700592,  # Sh_2212_lllvjj_Int
                    700590,  # Sh_2212_llvvjj_ss
                    700603,  # Sh_2212_llvv_ss
                    700589,  # Sh_2212_llvvjj_os
                    700602,  # Sh_2212_llvv_os
                    700709,  # Sh_2212_lvgammajj
                    700601,  # Sh_2212_lllv
                    700588,  # Sh_2212_lllvjj
                    700591,  # Sh_2212_lllljj_Int
                ],
            )
        case "dijet":
            return MCSample(
                name="dijet",
                dsids=[
                    364700,  # JZ0WithSW
                    364701,  # JZ1WithSW
                    364702,  # JZ2WithSW
                    364703,  # JZ3WithSW
                    364704,  # JZ4WithSW
                    364705,  # JZ5WithSW
                    364706,  # JZ6WithSW
                    364707,  # JZ7WithSW
                    364708,  # JZ8WithSW
                    364709,  # JZ9WithSW
                    364710,  # JZ10WithSW
                    364711,  # JZ11WithSW
                    364712,  # JZ12WithSW
                ],
            )
        case "dijet_bb":
            return MCSample(
                name="dijet_bb",
                dsids=[
                    802067,  # JZ1_4jets15_2bjets
                    800285,  # JZ2_4jets15_2bjets
                    800286,  # JZ3_4jets15_2bjets
                    800287,  # JZ4_4jets15_2bjets
                    802068,  # JZ5_4jets15_2bjets
                    802069,  # JZ6_4jets15_2bjets
                    802070,  # JZ7_4jets15_2bjets
                    802071,  # JZ8_4jets15_2bjets
                    802072,  # JZ9incl_4jets15_2bjets
                ],
            )
        case "multiboson":
            return MCSample(
                name="multiboson",
                dsids=[
                    364242,  # Sherpa 2.2.2 WWW 3l3v
                    364243,  # Sherpa 2.2.2 WWZ 4l2v
                    364244,  # Sherpa 2.2.2 WWZ 2l4v
                    364245,  # Sherpa 2.2.2 WZZ 5l1v
                    364246,  # Sherpa 2.2.2 WWZZ 3l3v
                    364247,  # Sherpa 2.2.2 ZZZ 6l0v
                    364248,  # Sherpa 2.2.2 ZZZ 4l2v
                    364249,  # Sherpa 2.2.2 ZZZ 2l4v
                ],
            )
        case "raretop":
            return MCSample(
                name="raretop",
                dsids=[
                    # 412043 # MadGraph+Pythia8 tttt
                    # 525662 # MadGraph+Pythia8 tttW
                    # 525663 # MadGraph+Pythia8 tttj
                    410081,  # MadGraph+Pythia8 ttWW
                    # 500463 # MadGraph+Pythia8 ttWZ
                    # 525359 # MadGraph+Pythia8 ttHH
                    # 525360 # MadGraph+Pythia8 ttWH
                    # 500462 # MadGraph+Pythia8 ttZZ
                    525955,  # MadGraph+Pythia8 tWZ
                ],
            )
        case "raretop_fast":
            return MCSample(
                name="raretop_fast",
                dsids=[
                    412043,  # MadGraph+Pythia8 tttt
                    525662,  # MadGraph+Pythia8 tttW
                    525663,  # MadGraph+Pythia8 tttj
                    500463,  # MadGraph+Pythia8 ttWZ
                    500462,  # MadGraph+Pythia8 ttZZ
                ],
            )
        case "singletop_dilepton_Wt_DR_dyn":
            return MCSample(
                name="singletop_dilepton_Wt_DR_dyn",
                dsids=[
                    601354,  # Powheg+Pythia8 Wt-chan dilepton, DR (top), dynamic scale
                    601353,  # Powheg+Pythia8 Wt-chan dilepton, DR (antitop), dynamic scale
                ],
            )
        case "singletop_dilepton_Wt_DS_dyn":
            return MCSample(
                name="singletop_dilepton_Wt_DS_dyn",
                dsids=[
                    601628,  # Powheg+Pythia8 Wt-chan dilepton, DS (top), dynamic scale
                    601624,  # Powheg+Pythia8 Wt-chan dilepton, DS (antitop), dynamic scale
                ],
            )
        case "singletop_inclusive_Wt_DR_dyn":
            return MCSample(
                name="singletop_inclusive_Wt_DR_dyn",
                dsids=[
                    601355,  # Powheg+Pythia8 Wt-chan inclusive, DR (top), dynamic scale
                    601352,  # Powheg+Pythia8 Wt-chan inclusive, DR (antitop), dynamic scale
                ],
            )
        case "singletop_inclusive_Wt_DS_dyn":
            return MCSample(
                name="singletop_inclusive_Wt_DS_dyn",
                dsids=[
                    601631,  # Powheg+Pythia8 Wt-chan inclusive, DS (top), dynamic scale
                    601627,  # Powheg+Pythia8 Wt-chan inclusive, DS (antitop), dynamic scale
                ],
            )
        case "singletop_inclusive_noWt":
            return MCSample(
                name="singletop_inclusive_noWt",
                dsids=[
                    410658,  # Powheg+Pythia8 t-chan (top)
                    410659,  # Powheg+Pythia8 t-chan (anititop)
                    410644,  # Powheg+Pythia8 s-chan (top)
                    410645,  # Powheg+Pythia8 s-chan (antitop)
                ],
            )
        case "ttH":
            return MCSample(
                name="ttH",
                dsids=[
                    346343,  # Powheg+Pythia8 ttH (allhad)
                    346344,  # Powheg+Pythia8 ttH (semilep)
                    346345,  # Powheg+Pythia8 ttH (dilep)
                ],
            )
        case "ttW":
            return MCSample(
                name="ttW",
                dsids=[
                    701260,  # Sherpa 2.2.14 ttW 0L
                    701261,  # Sherpa 2.2.14 ttW 1L
                    701262,  # Sherpa 2.2.14 ttW 2L
                ],
            )
        case "ttZ":
            return MCSample(
                name="ttZ",
                dsids=[
                    504330,  # aMcAtNlo+Pythia8 ttee
                    504334,  # aMcAtNlo+Pythia8 ttmumu
                    504338,  # aMcAtNlo+Pythia8 ttZqq
                    504342,  # aMcAtNlo+Pythia8 tttautau
                    504346,  # aMcAtNlo+Pythia8 ttZnunu
                ],
            )
        case "ttbar_bb":
            return MCSample(
                name="ttbar_bb",
                dsids=[
                    603003,  # PhPy8EG_A14_NNPDF31_ttbb_4FS_pTdef1_allhad_shower
                    603192,  # PhPy8_A14_NNPDF31_ttbb_4FS_bzd5_ljets_shower
                    603190,  # PhPy8_A14_NNPDF31_ttbb_4FS_bzd5_dilep_shower
                ],
            )
        case "ttbar_dilepton":
            return MCSample(
                name="ttbar_dilepton",
                dsids=[
                    410472,  # Powheg+Pythia8 dilepton
                ],
            )
        case "ttbar_inclusive":
            return MCSample(
                name="ttbar_inclusive",
                dsids=[
                    410470,  # Powheg+Pythia8 non-allhad
                    410471,  # Powheg+Pythia8 allhad
                ],
            )
        case "ttbar_inclusive_sliced_HT":
            return MCSample(
                name="ttbar_inclusive_sliced_HT",
                dsids=[
                    407342,  # Powheg+Pythia8 non-allhad HT1k5
                    407343,  # Powheg+Pythia8 non-allhad HT1k_1k5
                    407344,  # Powheg+Pythia8 non-allhad HT6c_1k
                ],
            )
        case "ttbar_inclusive_sliced_MET":
            return MCSample(
                name="ttbar_inclusive_sliced_MET",
                dsids=[
                    407345,  # Powheg+Pythia8 non-allhad MET200_300
                    407346,  # Powheg+Pythia8 non-allhad MET300_400
                    407347,  # Powheg+Pythia8 non-allhad MET400
                ],
            )

        case _:
            raise ValueError(f"Unknown process name: {name}")
