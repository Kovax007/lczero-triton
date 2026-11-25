"""Embedded attention-policy mapping-table module generation."""

import base64
import struct
import zlib

from lc0ex import SymbolArtifact
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import compile_ptx

_POLICY_SIZE = 1858
_POLICY_INPUT_SIZE = 4288
_SYMBOL_NAME = "lczero_bt4_mapping_table"
_BLACKWELL_ARCHITECTURE = 100
# ptxas refuses .version 8.7 for these targets; 8.8 (CUDA 12.9) assembles them.
# Every other target keeps its version, so older ptxas builds keep working.
_PTX_8_8_ARCHITECTURES = frozenset({103, 121})
_ENCODED_TABLE = (
    b"c-k%51(y)S8U|oe=>}oxl$Ms3?q0gPmhJ}W?(SH+yFpr7VnJG3VnJH^KAy`t^ZbJM%*^-BgbfoWOgO?5frxy^_e3HxQHVus;*gNU"
    b"q#zCH$wD^rke6R5LNSU{f|8V?G-W7DIm%Oos#GJ8I@G5LEoegrI@6Qh4B}UYFqB~oX9Ob|#c0MbmWfPaGE<n!Z050;WvpU7o7lx~"
    b"9N-{_Il@tnahwyJ<P@hl$9XPrkr4jm8aKJiL!R)8kAw+p5svUg<U7763Q>thbYc*PxWpqq2}nc|l9PfoWF{8{DMC?-QJfN#qBP~H"
    b"Kt(E1nLw&jgPPQ$E)8fx3tH2j?(}C6zcQF13}qN27|S@uGl8j0V>&aK!+aLAjFqfo1KZfgZyewthd9g;j&hnaoF#-yT;@8rxXVKx"
    b"^PCsF<6pjp{r@HiOE@AEg{VX$Ix&bzEMgOfgd`$>6r?6SnaM_O@=};06r~u&DMLBRQ-O+9qB2#eN;LwhO&#h|k0vyy4ejYn4|+3@"
    b"LHx>KhA@<ojA9I98OL}gFp)`2W(rf8$t-3whq)|fDXUn=CbqMS103WKhdILU9One5IL#T(a*p#{;36Sh;VRd-&JF(JE)RIZb6)ZB"
    b"zpNZu;fX**z9SNmi9%GO6N8w<AujQVPXYo+OcIikjFhC|N79p-Toj}@B`8TLN>hfil%qTqs6=G~sZI@QQj2=jrvVLVL{nPOhW2!#"
    b"JN+5TForXNk&I$AV;IXgCNPz0OlJmjna6w<u!Lo-VjUaV#t!yzgrgkecaC#{lbqr-X9(dEm$|`BZgG!?JmEPndB=Oce!FDC5{^hj"
    b"CJM2MLtNq!p9K6sLJ|=`3R04a^kgC%xyVZ)icpkd6sHVjDMxv#P>n#UQ-hk+qBeD?OFbIXgr+p34ejVmcX~5`LHx>KhA@<ojAArn"
    b"7|TQ^F@>p2V>&aK$t-3whq)|d5sO(uFsoS0CbqGQ{T$#Rhd9g;j&p*OoZ>WpaE|j_B!o*`<_cH2#&vFRn>*a)9{2g1Cp_a7@A&v%"
    b"t_FzkL?9yH5rwEkBRVmNMQq{_m-r;$2Leb;5|WaPRHP;iX-P*0GV>GJ$VEYlQ-YF|qBP~HKt(E1nJQGJ8i7=&Cbg(XeHze^Ml_>2"
    b"EoezA+R~oRbf*{n8Okt*GlH><V>}a>$Rs8+g{e$q26LImd=?PQQkJot)vRL^+t|rIj&PJ?oaPK?`Ga$u=K>cA;Sx8v$t~{lfQLNg"
    b"Ij?xf2fltA4#N|P$V4F)v57+>0!T~}l9G(%q#z}!NKXbbl8v0?r65HqK^e+Yj`CEYD%A+2Hg%~-eHze^Ml_}gO=(7J+R&CDI@68b"
    b"^k*=`7|AF`GlsEDWD=8^!c=B5i#g0?9`jkiLKd-@B?PmAm8@blYuLnAcCnA&IK)wobApqc;xy+t&jl_L!k=8>D%ZKeO>S|UJKW_S"
    b"_j$x)p74}V{^1pG`N&toDgY7qo+v~m8qtYC9O4p>_#_}9i3lJuNl8X>QjwZ8q$M30$wX$dkd^G@;%D+wkYbdiJQb)&B`On0b!t$P"
    b"TGXZvb*V>v8q$bnG^YhEX+;q2Xio<^(uMByrawa%&RE7Vo(W848q=A<OlC2gIm~4q3kYT@%UI4D*0PTEY-Ss~*vDaxahfxnC4@^{"
    b"=1;D0m1|t*1~<9S10E8}GoJIBcYNgQw~`luNJJ$Tv57-M5)nX3QjwZ8q$M3clAa7?BpcbuL0$?_l#-OC0#&F=H3F$k9qLk##x$iF"
    b"&1pePTG5&|v?Yj6bfybk=}kWdGn~<kV<MB7%oL_FlUdAW4s%(^B9;)$QkJot6|7_xt69SaHnNG$Y+)CBIm9u3=Okx1$9XPrkr1wM"
    b"m1|t*27hszJKW<w4|vEU9`l5!gz|!yyy7))_?M4-`Bvm35SeJiBo1+jM|=_xKw^@Rlw_nJC8<bFTGH_&8OcOuvXGS=<Rlll$wNL0"
    b"@(V>NNjWMKNOfvZlUmfHJ`HF{BO23$rZl5DEons%?PyO2I?|PHbf*VB=|g`8Gn_GuXDZW}&J5-<kNGTMA&Xed5`tOEa@Mexb*yI#"
    b"TiM2TcC(K|9OD#c3E>i#xxr0t@fWwb!(Hxip9h5UjOV=JE${fm*KdzC5s5@JViA{wBqD$mq$Cv?$VetKlZCAOL^iUMgS_M;KSe1;"
    b"St?SM>eQwVb*V>Vn$VPHw5BaVw4*&8=tw6z(}k||qBni$%V0(@n(<6#8Z(*2Z00bRg)Cw*O9*BKD_PAN*0PTEY+xgs*vuAou#;Wv"
    b"W)Fur$yxs30++bLRjzTJ8{Fm&ce%%X{^k*nc}ghHc+Lx6@`~5I;XNPt$R|D%A*yVNMr`8o0|6u^2}wytDpHe%w4@_F8OTT`vXGUZ"
    b"$U#nWk()f^rvL>hL}7|hlCo5!8a1d#eHze^Ml_>2EoezATGNKM1ksKTbfhcY=uQuM(wBbpX8;5FmEnwLJX4s?T;?&K1q8E{Wh`d}"
    b"D_O;A*07fKY+)<g*v=mIvXA{7<QOM8%SA48gPYvqJ`Z@v-#p?mPk2fw&v?UI-tn0)d?idYBQ}wUPAuY)ki?`QC8<bH1~QVJ9ONVy"
    b"xyi%N<Ru^ZDN1R|Qi-b6pbqtEOcR>YjMlWFEkSgm3tj0(cY4s1Ui799eHqMXCNP=l%w`@7S;S(N5X=fzvWnHLVFMf4%oet?jqU7U"
    b"C%f3q9u9Gm3tZ+;u5pvw+~F?wxX&XV^Mt2_@((X~$!p&5mUq1810VUsXCg!w8?lH-B9f4tRHP;iX-P*$GLe}qWF;Hf$w5wXlZT(l"
    b"PXP*2h{BYl3>B$L4eHW>W;CY-Eons%?PyO2I?{>GbfGKV=s{2V(vSWOU?77S&M3w+nd!`7KEW(y8OvG2TGp|i4Qyl+o7uuvwzG%5"
    b">|;L%IK~Oia)HZS=N9*Qz(YcL#&iDR1uuEUYu@md&wSx4VPfbuL?Q;Uh)+V2kb=~tCj%MDMs{+LkNgy%AcZK*FBGLLm8nWi>d=5D"
    b"G^aIfXiE^C=u8*7(u+Ryr62tnz(58wn#s&yHuG7`QdY2%Rjg(W8`#JuHnW8t>|{54*vmflbBL2%;2O91i@QAF5s!JoQ$l&cOJ4Ds"
    b"H~h<cKJbapeBmn*V%j&vBOyshNg946Bbmrd7P69qoa7=mdB{sX@>7696y_I7QjsdupbiaaN(+K$M|(QZk*;*3J3Z)0FM895zVu@N"
    b"0~yXp#xsfO%w|4ISjHOGvX1p^VJq9%&JK36i{0#DFZ=n8V;tu!=ef)^ZgG!?gz}8%yx}eH_?P#5;3J>-%$Jz||JVylIKmTwh$JKt"
    b"0VE~~Nl8X>Qjn8e<R%Y4lb3wtrvR0xOckn9jX<hXgPOFVEkU%SJss#sCpt5fVGL&kBN@eL#xRxzEMyUjSwb*NS;lg9v70^YWgq+b"
    b"AG8R~?E"
)


def values() -> tuple[int, ...]:
    """Return the ONNX gather inverse of LC0's attention-policy scatter map."""
    table = zlib.decompress(base64.b85decode(_ENCODED_TABLE))
    result = tuple(value[0] for value in struct.iter_unpack("<i", table))
    if (
        len(result) != _POLICY_SIZE
        or len(set(result)) != _POLICY_SIZE
        or min(result, default=-1) < 0
        or max(result, default=_POLICY_INPUT_SIZE) >= _POLICY_INPUT_SIZE
    ):
        message = "invalid embedded attention-policy mapping table"
        raise RuntimeError(message)
    return result


def compile_symbol(*, architecture: str) -> SymbolArtifact:
    """Compile the immutable attention-policy gather table as a module symbol."""
    entries = ", ".join(str(value) for value in values())
    architecture_number = int(architecture.removeprefix("sm_"))
    if architecture_number in _PTX_8_8_ARCHITECTURES:
        ptx_version = "8.8"
    else:
        ptx_version = "8.7" if architecture_number >= _BLACKWELL_ARCHITECTURE else "8.0"
    ptx = "\n".join(
        (
            f".version {ptx_version}",
            f".target {architecture}",
            ".address_size 64",
            "",
            f".visible .global .align 4 .u32 {_SYMBOL_NAME}[{_POLICY_SIZE}] = {{",
            f"  {entries}",
            "};",
            "",
        )
    )
    return SymbolArtifact(
        binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
        binary_data=compile_ptx(ptx, architecture=architecture),
        symbol_name=_SYMBOL_NAME,
    )
