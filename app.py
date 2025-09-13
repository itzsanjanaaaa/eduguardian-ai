import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import plotly.express as px


# Page config
st.set_page_config(
    page_title="EduGuardian AI", 
    page_icon="🎓", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
body { background-color: #f9fafb; }
.metric-card {
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 1rem; border-radius: 0.75rem; color: white; text-align: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 1rem;
}
.risk-high { background-color: #ff6b6b !important; }
.risk-medium { background-color: #f6ad55 !important; }
.risk-low { background-color: #48bb78 !important; }

/* Enhanced metric borders */
[data-testid="metric-container"] {
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.high-risk-metric {
    border: 3px solid #ff6b6b !important;
}
.medium-risk-metric {
    border: 3px solid #f6ad55 !important;
}
.low-risk-metric {
    border: 3px solid #48bb78 !important;
}
</style>
""", unsafe_allow_html=True)


# Load & preprocess
@st.cache_data
def load_data(path="dataset.csv"):
    df = pd.read_csv(path).dropna().reset_index(drop=True)
    df["Grade_diff"] = df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]
    df["Total_units_approved"] = df["Curricular units 1st sem (approved)"] + df["Curricular units 2nd sem (approved)"]
    df["Avg_grade"] = (df["Curricular units 1st sem (grade)"] + df["Curricular units 2nd sem (grade)"]) / 2
    df["Performance_ratio"] = df["Avg_grade"] / (df["Total_units_approved"] + 0.1)
    df["Financial_risk"] = (df["Tuition fees up to date"] == 0).astype(int) + df["Debtor"]
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"].astype(str))
    df["Scholarship holder"] = le.fit_transform(df["Scholarship holder"].astype(str))
    df["Debtor"] = le.fit_transform(df["Debtor"].astype(str))
    X = df[[
        "Age at enrollment",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
        "Tuition fees up to date",
        "Scholarship holder",
        "Debtor",
        "Gender",
        "Grade_diff",
        "Total_units_approved",
        "Avg_grade",
        "Performance_ratio",
        "Financial_risk"
    ]]
    y = df["Target"]
    return X, y, df


@st.cache_resource
def train_model(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )
    scaler = StandardScaler().fit(X_train)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(scaler.transform(X_train), y_train)
    return model, scaler


X, y, df_full = load_data()
model, scaler = train_model(X, y)
mapping = {"Dropout": "High Risk", "Enrolled": "Medium Risk", "Graduate": "Low Risk"}


# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("", [
    "🏠 Dashboard",
    "👨‍🎓 Student Analysis",
    "💬 Counselling Hub",
    "📱 Alerts",
    "📈 Analytics"
])


if page == "🏠 Dashboard":
    st.title("🎓 Dashboard")
    # Metrics
    acc = model.score(scaler.transform(X), y)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><h4>Model Accuracy</h4><h2>{acc:.1%}</h2></div>', unsafe_allow_html=True)
    with col2:
        total = len(df_full)
        st.markdown(f'<div class="metric-card"><h4>Total Students</h4><h2>{total:,}</h2></div>', unsafe_allow_html=True)
    with col3:
        high = (df_full["Financial_risk"]>=2).sum()
        st.markdown(f'<div class="metric-card risk-high"><h4>High Risk</h4><h2>{high:,}</h2></div>', unsafe_allow_html=True)
    # Risk distribution pie
    st.subheader("Risk Distribution")
    preds = model.predict(scaler.transform(X))
    df_full["Predicted Risk"] = pd.Series(preds).map(mapping)
    fig = px.pie(df_full, names="Predicted Risk", color="Predicted Risk",
                 color_discrete_map={"High Risk":"#ff6b6b","Medium Risk":"#f6ad55","Low Risk":"#48bb78"})
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


elif page == "👨‍🎓 Student Analysis":
    st.title("🔮 Individual Assessment")
    with st.form("f"):
        a = st.slider("Age at Enrollment", 17, 50, 20)
        u1 = st.slider("1st Sem Units Approved", 0, 20, 6)
        g1 = st.slider("1st Sem Grade", 0.0, 20.0, 12.0)
        u2 = st.slider("2nd Sem Units Approved", 0, 20, 6)
        g2 = st.slider("2nd Sem Grade", 0.0, 20.0, 12.0)
        tpd = st.selectbox("Tuition Paid?", ["✅ Yes","❌ No"])
        sch = st.selectbox("Scholarship?", ["✅ Yes","❌ No"])
        db = st.selectbox("Debtor?", ["⚠️ Yes","✅ No"])
        gen = st.selectbox("Gender", ["👨 Male","👩 Female"])
        submit = st.form_submit_button("🔍 Analyze")
    if submit:
        vals = [
            a,u1,u2,g1,g2,
            1 if tpd=="✅ Yes" else 0,
            1 if sch=="✅ Yes" else 0,
            1 if db=="⚠️ Yes" else 0,
            1 if gen=="👨 Male" else 0,
            g2-g1, u1+u2, (g1+g2)/2,
            ((g1+g2)/2)/(u1+u2+0.1),
            (0 if tpd=="✅ Yes" else 1)+(1 if db=="⚠️ Yes" else 0)
        ]
        X_in = pd.DataFrame([vals], columns=X.columns)
        p = model.predict(scaler.transform(X_in))[0]
        conf = model.predict_proba(scaler.transform(X_in)).max()
        lvl = mapping[p]

        # Enhanced risk display mapping
        risk_map = {
            "High Risk": {"color": "inverse", "emoji": "⚠️", "desc": "Immediate intervention recommended"},
            "Medium Risk": {"color": "normal", "emoji": "⚠️", "desc": "Monitor regularly"},
            "Low Risk": {"color": "off", "emoji": "✅", "desc": "On track for success"}
        }

        color = risk_map[lvl]["color"]
        emoji = risk_map[lvl]["emoji"]
        desc = risk_map[lvl]["desc"]

        # Apply class for border color styling
        metric_container_class = {
            "High Risk": "high-risk-metric",
            "Medium Risk": "medium-risk-metric",
            "Low Risk": "low-risk-metric"
        }[lvl]

        st.markdown(f'<div class="{metric_container_class}">', unsafe_allow_html=True)
        st.metric(f"Risk Level {emoji}", lvl, delta=f"{conf:.1%}", delta_color=color)
        st.markdown(f"</div>", unsafe_allow_html=True)

        # Detailed interpretation below metric
        st.markdown(f"**Interpretation:** {desc}")

        # Confidence message based on thresholds
        confidence_thresholds = {
            "High": 0.75,
            "Medium": 0.5,
            "Low": 0.25
        }

        if conf >= confidence_thresholds["High"]:
            conf_msg = "Model confident in prediction."
        elif conf >= confidence_thresholds["Medium"]:
            conf_msg = "Moderate confidence in prediction."
        else:
            conf_msg = "Low confidence, interpret cautiously."

        st.info(conf_msg)


elif page == "💬 Counselling Hub":
    st.title("💬 Counselling Hub")
    lvl = st.selectbox("Risk Category", ["High Risk","Medium Risk","Low Risk"])
    actions = {
        "High Risk": ["Schedule urgent session","Contact guardian","Financial plan","Academic tutoring","Mental health check"],
        "Medium Risk": ["Set meeting","Peer tutoring","Progress tracking","Goal plan"],
        "Low Risk": ["Celebrate","Leadership roles","Advanced goals","Mentoring"]
    }[lvl]
    st.markdown(f'<div class="metric-card {"risk-high" if lvl=="High Risk" else "risk-medium" if lvl=="Medium Risk" else "risk-low"}"><h3>{lvl} Intervention</h3></div>', unsafe_allow_html=True)
    for act in actions: st.write(f"• {act}")


elif page == "📱 Alerts":
    st.title("📱 Alert System")
    df_alert = pd.DataFrame({
        "Time": ["10:30","09:45","09:20","08:55","08:30"],
        "Student": ["STU001","STU002","STU003","STU004","STU005"],
        "Risk": ["High Risk","Medium Risk","High Risk","Medium Risk","Low Risk"],
        "Type": ["Acad Decline","Attendance","Financial","Grade Drop","Congrats"],
        "Status": ["Sent","Delivered","Pending","Delivered","Sent"]
    })
    def style_status(r):
        return f"background-color:{'#ff6b6b' if r=='High Risk' else '#f6ad55' if r=='Medium Risk' else '#48bb78'};color:white"
    st.dataframe(df_alert.style.applymap(style_status, subset=["Risk"]))


elif page == "📈 Analytics":
    st.title("📈 Analytics Dashboard")
    months = ["Jan","Feb","Mar","Apr","May","Jun"]
    hr = [45,38,42,35,28,23]
    mr = [120,115,108,95,78,67]
    fig = px.line(x=months, y=[hr,mr], labels={"x":"Month","value":"Count"}, color_discrete_map={"wide_variable_0":"#ff6b6b","wide_variable_1":"#f6ad55"})
    fig.update_layout(legend=dict(title="Risk", orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Intervention Success Rates")
    df_int = pd.DataFrame({
        "Intervention":["Tutoring","Financial Aid","Counselling","Peer Support","Mentoring"],
        "Success":[85,92,78,88,95]
    })
    fig2 = px.bar(df_int, x="Success", y="Intervention", orientation="h", color="Success", color_continuous_scale="greens")
    st.plotly_chart(fig2, use_container_width=True)
