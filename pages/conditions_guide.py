import streamlit as st
from pathlib import Path
import streamlit.components.v1 as components
st.set_page_config(page_title="SkinAI — Conditions Guide", page_icon="🧬", layout="wide")

# ...existing code...
st.markdown("""
<style>
/* remove top chrome + top gap (current Streamlit selectors) */
header[data-testid="stHeader"]{
  display:none !important;
  height:0 !important;
}
div[data-testid="stToolbar"]{
  display:none !important;
  height:0 !important;
}
div[data-testid="stDecoration"]{
  display:none !important;
  height:0 !important;
}
section[data-testid="stMain"]{
  padding-top:0 !important;
}
div[data-testid="stMainBlockContainer"]{
  padding-top:0 !important;
  margin-top:0 !important;
}

/* force navbar to stick to top */
.skinai-nav{
  margin-top:0 !important;
}
:root {
  --bg:#07101f; --card:rgba(10,20,45,0.72); --border:rgba(56,189,248,0.15);
  --text:#e2e8f0; --muted:#94a3b8;
}

/* app background */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background: var(--bg) !important;
  color: var(--text) !important;
}

/* remove header + ALL top padding */
[data-testid="stToolbar"], [data-testid="stHeader"] {
  display: none !important;
  height: 0 !important;
}
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] > .main > div,
[data-testid="stAppViewContainer"] > .main .block-container {
  padding-top: 0 !important;
  margin-top: 0 !important;
}

/* cards/content */
.card {
  background:var(--card);
  border:1px solid var(--border);
  border-radius:16px;
  padding:16px;
  margin-bottom:16px;
}
.title {
  font-size:2rem; font-weight:900;
  background:linear-gradient(90deg,#38bdf8,#818cf8,#c084fc);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
img {border-radius:12px; border:1px solid var(--border);}
.skinai-nav {
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 9999;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 18px 50px;
  background: rgba(10, 20, 45, 0.6);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  border-bottom: 1px solid rgba(56, 189, 248, 0.12);
}

/* IMPORTANT: push page content below fixed navbar */
section[data-testid="stMain"] {
  padding-top: 90px !important;
}
div[data-testid="stMainBlockContainer"] {
  padding-top: 0 !important;
  margin-top: 0 !important;
}
.nav-logo-brand { font-size: 26px; font-weight: 800; color: #e2e8f0; letter-spacing: 1px; }
.nav-logo-skin {
  font-weight: 900; font-size: 28px;
  background: linear-gradient(90deg, #38bdf8, #818cf8);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.nav-links-area { display: flex; gap: 0; }
.nav-links-area a {
  text-decoration: none; color: #94a3b8;
  font-weight: 600; font-size: 15px; margin-left: 32px;
  transition: color 0.3s; position: relative;
}
.nav-links-area a::after {
  content: ''; position: absolute; left: 0; bottom: -3px;
  width: 0; height: 2px;
  background: linear-gradient(90deg, #38bdf8, #818cf8);
  transition: width 0.3s ease; border-radius: 1px;
}
.nav-links-area a:hover { color: #e2e8f0; }

# /* navbar */
# .skinai-nav{
#   position: sticky;
#   top: 0;
#   z-index: 999;
#   display:flex;
#   align-items:center;
#   justify-content:space-between;

#   margin: 0 0 18px 0 !important;   /* no top gap */
#   padding: 12px 18px;

#   border: none;
#   border-bottom: 1px solid var(--border);
#   border-radius: 0;                /* remove rounded box */
#   background: rgba(10,20,45,0.72);
#   backdrop-filter: blur(8px);

#   /* stretch across container width like header strip */
#   width: calc(100% + 2rem);
#   margin-left: -1rem;
#   margin-right: -1rem;
# }
# .nav-left{display:flex; align-items:center; gap:10px;}
# .nav-logo-brand{font-weight:800; color:#e2e8f0;}
# .nav-logo-skin{
#   background: linear-gradient(90deg,#38bdf8,#818cf8,#c084fc);
#   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
# }
# .nav-links-area{display:flex; gap:16px;}
# .nav-links-area a{color:#9ca3af; text-decoration:none; font-weight:600;}
# .nav-links-area a:hover{color:#e2e8f0;}
#skinai-hamburger{background:none; border:none; color:#9ca3af; font-size:22px; cursor:pointer;}
.info-card{
  display:flex;
  gap:20px;
  align-items:flex-start;
  background:var(--card);
  border:1px solid var(--border);
  border-radius:16px;
  padding:16px;
  margin-bottom:16px;
}
.info-card .media{
  flex:0 0 320px;
}
.info-card .media img{
  width:100%;
  height:auto;
  display:block;
  border-radius:12px;
  border:1px solid var(--border);
}
.info-card .content{
  flex:1;
}
@media (max-width: 900px){
  .info-card{flex-direction:column;}
  .info-card .media{flex:1 1 auto; width:100%;}
}
</style>
""", unsafe_allow_html=True)


ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "assets" / "conditions"



st.markdown("""
<nav class="skinai-nav">
  <div class="nav-left">
    <span class="nav-logo-brand"><span class="nav-logo-skin">Skin</span>AI</span>
  </div>
  <div class="nav-links-area">
    <a href="/" target="_self">Home</a>
  </div>
</nav>
""", unsafe_allow_html=True)

components.html("""
<script>
(function () {
    function attach() {
        var btn = window.parent.document.getElementById('skinai-hamburger');
        if (!btn) { setTimeout(attach, 120); return; }
        btn.addEventListener('click', function () {
            var close = window.parent.document.querySelector('[data-testid="stSidebarCollapseButton"] button');
            if (close) { close.click(); return; }
            var open = window.parent.document.querySelector('[data-testid="collapsedControl"]');
            if (open) { open.click(); }
        });
    }
    attach();
})();
</script>
""", height=0)

DATA = [
    ("akiec", "Actinic Keratosis",
     "Actinic Keratosis is a rough, scaly lesion caused by prolonged exposure to ultraviolet (UV) radiation, commonly seen in older individuals or people with significant sun exposure. It usually appears on sun-exposed areas such as the face, scalp, ears, neck, and hands. These lesions may feel dry or crusty and can be skin-colored, red, or brown. Although it is not cancer itself, it is considered a premalignant condition because it can progress into squamous cell carcinoma if untreated. Early diagnosis and treatment, such as cryotherapy or topical medications, are important to prevent malignant transformation."),

    ("bcc", "Basal Cell Carcinoma",
     "Basal Cell Carcinoma is the most common type of skin cancer and arises from the basal cells in the epidermis. It is strongly associated with long-term UV exposure and is more frequent in fair-skinned individuals. BCC typically appears as a pearly or translucent bump, sometimes with visible blood vessels (telangiectasia), or as a non-healing ulcer. It grows slowly and rarely metastasizes, but it can invade surrounding tissues if neglected. Early treatment, including surgical excision or topical therapies, leads to excellent prognosis."),

    ("bkl", "Benign Keratosis",
     "Benign Keratosis includes non-cancerous skin growths such as seborrheic keratosis. These lesions are very common, especially in older adults, and are not related to sun exposure. They often appear as well-defined, waxy, or 'stuck-on' growths with colors ranging from light tan to dark brown or black. They may have a rough or verrucous surface. These lesions are harmless and do not become cancerous, but they can sometimes be mistaken for melanoma, so proper diagnosis is important."),

    ("df", "Dermatofibroma",
     "Dermatofibroma is a benign fibrous skin nodule that often develops after minor skin trauma such as insect bites, splinters, or cuts. It commonly appears on the legs and is more frequent in young adults. The lesion is usually firm to the touch and may be pink, brown, or reddish. A characteristic feature is the 'dimple sign,' where the lesion dimples inward when pinched. Dermatofibromas are harmless, stable, and do not require treatment unless they cause discomfort or cosmetic concern."),

    ("mel", "Melanoma",
     "Melanoma is a highly aggressive form of skin cancer that originates from melanocytes, the pigment-producing cells of the skin. Although less common than other skin cancers, it is responsible for the majority of skin cancer-related deaths due to its high potential to metastasize. It can develop from an existing mole or appear as a new lesion. Key warning signs are summarized by the ABCDE rule: Asymmetry, irregular Borders, multiple Colors, Diameter greater than 6mm, and Evolving changes. Early detection is critical, as prognosis significantly worsens once the cancer spreads."),

    ("nv", "Melanocytic Nevus",
     "Melanocytic Nevus, commonly known as a mole, is a benign proliferation of melanocytes. These are very common and can be present from birth or develop over time. They are usually round or oval, with a uniform color and smooth borders. While most moles are harmless, some can transform into melanoma, especially if they show changes in size, shape, or color. Regular monitoring using the ABCDE criteria is recommended, particularly for individuals with many moles or a family history of melanoma."),

    ("vasc", "Vascular Lesion",
     "Vascular lesions are abnormalities of blood vessels in the skin, including conditions such as hemangiomas, angiomas, and vascular malformations. These lesions typically appear red, purple, or blue due to the presence of blood vessels. Most vascular lesions are benign and may be present from birth or develop later in life. They are usually harmless and do not require treatment unless they bleed, grow rapidly, or cause cosmetic or functional concerns. Some may fade over time, especially infantile hemangiomas."),
]


import base64

def to_data_uri(path: Path) -> str:
    ext = path.suffix.lower().replace(".", "") or "jpg"
    mime = "jpeg" if ext == "jpg" else ext
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"

for code, name, desc in DATA:
    img_path = IMG_DIR / f"{code}.jpg"
    if img_path.exists():
        img_uri = to_data_uri(img_path)
        img_html = f'<img src="{img_uri}" alt="{name}"/>'
    else:
        img_html = f'<div style="color:var(--muted);">Add image: assets/conditions/{code}.jpg</div>'

    st.markdown(
        f"""
        <div class="info-card">
          <div class="media">
            {img_html}
          </div>
          <div class="content">
            <h3>{name}</h3>
            <p style="color:var(--muted); margin:0;">{desc}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )