import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LinearRegression

# Appliquer un style global aux graphiques
plt.style.use("ggplot")

# ================================
# Titre et Description de l'Application
# ================================
st.set_page_config(page_title="Estimation de la Valeur au Marché", layout="wide")
st.title("Application Avancée d'Estimation de la Valeur au Marché d'une Entreprise")
st.markdown("""
Cette application vous permet d'estimer la valeur au marché d'une entreprise en combinant plusieurs méthodes.  
Elle offre les fonctionnalités suivantes :
- Personnalisation des pondérations pour chaque méthode d’évaluation.
- Visualisations graphiques améliorées.
- Calcul de la valeur intrinsèque via la méthode DCF.
- Comparaison des ratios aux normes industrielles et seuils minimaux.
- Prévisions automatiques du cours de l’action et recommandations.
- Export des résultats.
""")
st.markdown("---")

# ================================
# Configuration dans la Barre Latérale
# ================================
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Entrez le ticker de l'entreprise (ex: AAPL, MSFT)", "AAPL")
if st.sidebar.button("Rafraîchir les données"):
    st.cache_data.clear()  # Efface le cache pour récupérer de nouvelles données
    st.success("Les données ont été rafraîchies.")

st.sidebar.header("Normes de l'Industrie")
industry_pe = st.sidebar.number_input("Norme P/E", value=15.0, format="%.2f", help="Norme moyenne du P/E dans l'industrie")
industry_dividend_yield = st.sidebar.number_input("Norme Dividend Yield", value=0.03, format="%.3f", help="Norme moyenne du Dividend Yield dans l'industrie")
industry_ev_ebitda = st.sidebar.number_input("Norme EV/EBITDA", value=10.0, format="%.2f", help="Norme moyenne du EV/EBITDA dans l'industrie")
industry_fcf_yield = st.sidebar.number_input("Norme Free Cash Flow Yield", value=0.10, format="%.3f", help="Norme moyenne du Free Cash Flow Yield dans l'industrie")
industry_net_margin = st.sidebar.number_input("Norme Marge Nette", value=0.10, format="%.3f", help="Norme moyenne de la Marge Nette dans l'industrie")
industry_debt_ratio = st.sidebar.number_input("Norme Debt Ratio", value=0.50, format="%.2f", help="Norme moyenne du Debt Ratio dans l'industrie")

st.sidebar.header("Seuils Minimum Acceptables")
min_pe_threshold = st.sidebar.number_input("Seuil minimum acceptable P/E", value=10.0, format="%.2f", help="Seuil minimum acceptable pour le ratio P/E")
min_dividend_yield_threshold = st.sidebar.number_input("Seuil minimum acceptable Dividend Yield", value=0.02, format="%.3f", help="Seuil minimum acceptable pour le Dividend Yield")
min_ev_ebitda_threshold = st.sidebar.number_input("Seuil minimum acceptable EV/EBITDA", value=8.0, format="%.2f", help="Seuil minimum acceptable pour le EV/EBITDA")
min_fcf_yield_threshold = st.sidebar.number_input("Seuil minimum acceptable Free Cash Flow Yield", value=0.08, format="%.3f", help="Seuil minimum acceptable pour le Free Cash Flow Yield")
min_net_margin_threshold = st.sidebar.number_input("Seuil minimum acceptable Marge Nette", value=0.05, format="%.3f", help="Seuil minimum acceptable pour la Marge Nette")
min_debt_ratio_threshold = st.sidebar.number_input("Seuil minimum acceptable Debt Ratio", value=0.30, format="%.2f", help="Seuil minimum acceptable pour le Debt Ratio")

st.sidebar.header("Pondération des Méthodes")
weight_pe = st.sidebar.slider("Pondération pour P/E", 0.0, 1.0, 0.2, help="Importance de la méthode P/E")
weight_dividend = st.sidebar.slider("Pondération pour Dividende", 0.0, 1.0, 0.2, help="Importance de la méthode Dividende")
weight_ebitda = st.sidebar.slider("Pondération pour EBITDA", 0.0, 1.0, 0.2, help="Importance de la méthode EBITDA")
weight_fcf = st.sidebar.slider("Pondération pour Free Cash Flow", 0.0, 1.0, 0.2, help="Importance de la méthode Free Cash Flow")
weight_margin = st.sidebar.slider("Pondération pour Marge Nette", 0.0, 1.0, 0.2, help="Importance de la méthode Marge Nette")

# ================================
# Description de la Compagnie et du Secteur
# ================================
st.header("Description de la Compagnie et du Secteur")
try:
    stock = yf.Ticker(ticker_input)
    info = stock.info
    description = info.get("longBusinessSummary", "Description non disponible.")
    industry = info.get("industry", "Industrie non disponible.")
    sector = info.get("sector", "Secteur non disponible.")
    
    st.markdown(f"**Secteur :** {sector}")
    st.markdown(f"**Industrie :** {industry}")
    st.markdown("**Description :**")
    st.write(description)
except Exception as e:
    st.error("Erreur lors de la récupération de la description de la compagnie : " + str(e))
st.markdown("---")

# ================================
# Récupération des Données Financières
# ================================
@st.cache_data(show_spinner=False)
def get_financial_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        shares_outstanding = info.get("sharesOutstanding", None)
        pe_ratio = info.get("trailingPE", None)
        eps = info.get("trailingEps", None)
        dividend_yield = info.get("dividendYield", None)
        dividend_rate = info.get("dividendRate", None)
        ebitda = info.get("ebitda", None)
        ev_ebitda = info.get("enterpriseToEbitda", None)
        free_cash_flow = info.get("freeCashflow", None)
        revenue = info.get("totalRevenue", None)
        net_margin = info.get("profitMargins", None)
        total_debt = info.get("totalDebt", None)
        total_assets = info.get("totalAssets", None)
        
        debt_ratio = None
        if total_debt and total_assets:
            debt_ratio = total_debt / total_assets

        return {
            "shares_outstanding": shares_outstanding,
            "pe_ratio": pe_ratio,
            "eps": eps,
            "dividend_yield": dividend_yield,
            "dividend_rate": dividend_rate,
            "ebitda": ebitda,
            "ev_ebitda": ev_ebitda,
            "free_cash_flow": free_cash_flow,
            "revenue": revenue,
            "net_margin": net_margin,
            "debt_ratio": debt_ratio
        }
    except Exception as e:
        st.error("Erreur lors de la récupération des données : " + str(e))
        return {}

with st.spinner("Récupération des données financières..."):
    data = get_financial_data(ticker_input)

# ================================
# Saisie et Modification des Données Financières
# ================================
st.header("1. Données Financières (récupérées et modifiables)")
col1, col2 = st.columns(2)
with col1:
    user_pe = st.number_input("Ratio cours/bénéfice (P/E)",
                                value=data.get("pe_ratio") if data.get("pe_ratio") is not None else 0.0,
                                format="%.2f",
                                help="Multiplicateur indiquant combien l'action se vend à X fois ses bénéfices.")
    user_net_margin = st.number_input("Marge nette (ex: 0.15 pour 15%)",
                                      value=data.get("net_margin") if data.get("net_margin") is not None else 0.0,
                                      format="%.3f",
                                      help="Pourcentage de profit réalisé sur le chiffre d'affaires.")
    user_dividend_yield = st.number_input("Dividend Yield (ex: 0.03 pour 3%)",
                                          value=data.get("dividend_yield") if data.get("dividend_yield") is not None else 0.0,
                                          format="%.3f",
                                          help="Rendement des dividendes par rapport au prix de l'action.")
with col2:
    user_eps = st.number_input("Bénéfice par action (EPS)",
                               value=data.get("eps") if data.get("eps") is not None else 0.0,
                               format="%.2f",
                               help="Bénéfice net par action.")
    user_ebitda = st.number_input("EBITDA",
                                  value=data.get("ebitda") if data.get("ebitda") is not None else 0.0,
                                  format="%.2f",
                                  help="Bénéfice avant intérêts, impôts, dépréciation et amortissement.")
    user_free_cash_flow = st.number_input("Free Cash Flow",
                                          value=data.get("free_cash_flow") if data.get("free_cash_flow") is not None else 0.0,
                                          format="%.2f",
                                          help="Flux de trésorerie disponible pour l'entreprise.")

st.markdown("### Valeurs Optionnelles Complémentaires")
user_ev_ebitda = st.number_input("Multiple EV/EBITDA",
                                 value=data.get("ev_ebitda") if data.get("ev_ebitda") is not None else 0.0,
                                 format="%.2f",
                                 help="Multiple basé sur l'EBITDA pour évaluer la valeur d'entreprise.")
user_fcf_yield = st.number_input("Rendement du Free Cash Flow (ex: 0.10 pour 10%)",
                                 value=0.0,
                                 format="%.3f",
                                 help="Rendement attendu du flux de trésorerie disponible.")
st.markdown("> **Note :** Les valeurs par défaut proviennent de Yahoo Finance. Vous pouvez les ajuster manuellement si nécessaire.")
st.markdown("---")

# ================================
# Estimation de la Valeur au Marché
# ================================
st.header("3. Estimation de la Valeur au Marché")

def compute_estimates(user_pe, user_eps, user_dividend_yield, data, 
                      user_ebitda, user_ev_ebitda, user_free_cash_flow, user_fcf_yield, user_net_margin):
    estimates = {}
    shares_outstanding = data.get("shares_outstanding")
    revenue = data.get("revenue")
    
    if user_pe > 0 and user_eps > 0 and shares_outstanding:
        est_pe = user_pe * user_eps * shares_outstanding
        estimates["P/E"] = est_pe
    
    if user_dividend_yield > 0 and data.get("dividend_rate") and shares_outstanding:
        dividend_rate = data.get("dividend_rate")
        est_div = (dividend_rate / user_dividend_yield) * shares_outstanding
        estimates["Dividende"] = est_div
    
    if user_ebitda > 0 and user_ev_ebitda > 0:
        est_ebitda = user_ebitda * user_ev_ebitda
        estimates["EBITDA"] = est_ebitda
    
    if user_free_cash_flow > 0 and user_fcf_yield > 0:
        est_fcf = user_free_cash_flow / user_fcf_yield
        estimates["Free Cash Flow"] = est_fcf
    
    if user_net_margin > 0 and revenue and user_pe > 0:
        net_income = revenue * user_net_margin
        est_margin = net_income * user_pe
        estimates["Marge Nette"] = est_margin
    
    return estimates

estimates = compute_estimates(user_pe, user_eps, user_dividend_yield, data,
                              user_ebitda, user_ev_ebitda, user_free_cash_flow, user_fcf_yield,
                              user_net_margin)

total_weight = weight_pe + weight_dividend + weight_ebitda + weight_fcf + weight_margin
if total_weight == 0:
    total_weight = 1

weighted_estimates = {}
for method, est in estimates.items():
    if method == "P/E":
        weighted_estimates[method] = est * (weight_pe / total_weight)
    elif method == "Dividende":
        weighted_estimates[method] = est * (weight_dividend / total_weight)
    elif method == "EBITDA":
        weighted_estimates[method] = est * (weight_ebitda / total_weight)
    elif method == "Free Cash Flow":
        weighted_estimates[method] = est * (weight_fcf / total_weight)
    elif method == "Marge Nette":
        weighted_estimates[method] = est * (weight_margin / total_weight)

if st.button("Calculer la valeur au marché"):
    if estimates:
        final_estimate = sum(weighted_estimates.values())
        st.subheader("Estimation de la Valeur au Marché")
        st.write("Estimation basée sur les méthodes pondérées :")
        st.write(f"**{final_estimate:,.2f}**")
        
        # Affichage détaillé des estimations avec descriptions
        ratio_descriptions = {
            "P/E": "Compare le prix de l'action aux bénéfices par action.",
            "Dividende": "Basé sur le rendement du dividende.",
            "EBITDA": "Utilise l'EBITDA comme mesure de performance opérationnelle.",
            "Free Cash Flow": "Estime la valeur en fonction du Free Cash Flow disponible.",
            "Marge Nette": "Calcule la valeur en multipliant le bénéfice net par le P/E."
        }
        
        st.markdown("#### Détail des estimations (non pondérées)")
        for method, value in estimates.items():
            st.write(f"- **{method}** : {value:,.2f}")
            st.write(f"  *{ratio_descriptions.get(method, 'Pas de description disponible.')}*")
        
        st.markdown("#### Pondérations utilisées")
        st.write(f"P/E: {weight_pe:.2f}, Dividende: {weight_dividend:.2f}, EBITDA: {weight_ebitda:.2f}, Free Cash Flow: {weight_fcf:.2f}, Marge Nette: {weight_margin:.2f}")
        
        # Graphique des contributions pondérées amélioré
        fig, ax = plt.subplots(figsize=(10, 6))
        methods_list = list(weighted_estimates.keys())
        contributions = list(weighted_estimates.values())
        bars = ax.bar(methods_list, contributions, edgecolor="grey")
        ax.set_xlabel("Méthodes", fontsize=12)
        ax.set_ylabel("Contribution pondérée", fontsize=12)
        ax.set_title("Contribution de chaque méthode à l'estimation finale", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:,.0f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Analyse Historique du Market Cap
        st.markdown("### Analyse Historique du Market Cap")
        if data.get("shares_outstanding"):
            try:
                stock = yf.Ticker(ticker_input)
                hist = stock.history(period="max")
                hist['MarketCap'] = hist['Close'] * data.get("shares_outstanding")
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ax2.plot(hist.index, hist['MarketCap'], linewidth=2)
                ax2.set_xlabel("Date", fontsize=12)
                ax2.set_ylabel("Market Cap", fontsize=12)
                ax2.set_title("Évolution Historique du Market Cap", fontsize=14, fontweight="bold")
                ax2.grid(True, linestyle="--", alpha=0.7)
                fig2.autofmt_xdate()
                plt.tight_layout()
                st.pyplot(fig2)
            except Exception as e:
                st.error("Erreur lors de la récupération des données historiques : " + str(e))
        else:
            st.info("Les données sur le nombre d'actions en circulation ne sont pas disponibles pour l'analyse historique.")
        
        # Export CSV des résultats
        results = {
            "Méthode": list(estimates.keys()),
            "Estimation non pondérée": list(estimates.values()),
            "Pondération": [weight_pe if m=="P/E" else weight_dividend if m=="Dividende" 
                             else weight_ebitda if m=="EBITDA" else weight_fcf if m=="Free Cash Flow" 
                             else weight_margin if m=="Marge Nette" else 0 for m in estimates.keys()],
            "Contribution pondérée": [weighted_estimates[m] for m in estimates.keys()]
        }
        results_df = pd.DataFrame(results)
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger les résultats en CSV", data=csv, file_name="resultats_estimation.csv", mime="text/csv")
    else:
        st.error("Pas assez de données disponibles pour effectuer une estimation.")
st.markdown("---")

# ================================
# Calcul de la Valeur Intrinsèque (DCF)
# ================================
st.header("4. Calcul de la Valeur Intrinsèque de l'Entreprise (DCF)")
st.markdown("""
Cette section utilise la méthode d'actualisation des flux de trésorerie (Discounted Cash Flow, DCF) pour estimer la valeur intrinsèque de l'entreprise.  
Définissez les paramètres suivants :
- **Nombre d'années de prévision**
- **Taux de croissance des FCF**
- **Taux d'actualisation**
- **Taux de croissance terminal**
""")
col_dcf1, col_dcf2, col_dcf3 = st.columns(3)
with col_dcf1:
    dcf_years = st.number_input("Nombre d'années de prévision", min_value=1, max_value=20, value=5, step=1,
                                help="Nombre d'années pour estimer les flux futurs")
with col_dcf2:
    growth_rate = st.number_input("Taux de croissance des FCF (%)", value=5.0, format="%.2f",
                                  help="Taux de croissance annuel prévu pour les Free Cash Flow")
with col_dcf3:
    discount_rate = st.number_input("Taux d'actualisation (%)", value=10.0, format="%.2f",
                                    help="Taux d'actualisation pour actualiser les flux futurs")
terminal_growth_rate = st.number_input("Taux de croissance terminal (%)", value=3.0, format="%.2f",
                                       help="Taux de croissance appliqué pour estimer la valeur terminale")

if st.button("Calculer la valeur intrinsèque"):
    if user_free_cash_flow > 0:
        growth_rate_dec = growth_rate / 100.0
        discount_rate_dec = discount_rate / 100.0
        terminal_growth_rate_dec = terminal_growth_rate / 100.0
        
        dcf_value = 0
        for i in range(1, int(dcf_years) + 1):
            fcf_forecast = user_free_cash_flow * ((1 + growth_rate_dec) ** i)
            dcf_value += fcf_forecast / ((1 + discount_rate_dec) ** i)
        
        fcf_final = user_free_cash_flow * ((1 + growth_rate_dec) ** dcf_years)
        terminal_value = fcf_final * (1 + terminal_growth_rate_dec) / (discount_rate_dec - terminal_growth_rate_dec)
        terminal_value_discounted = terminal_value / ((1 + discount_rate_dec) ** dcf_years)
        
        intrinsic_value = dcf_value + terminal_value_discounted
        
        intrinsic_value_per_share = None
        if data.get("shares_outstanding"):
            intrinsic_value_per_share = intrinsic_value / data.get("shares_outstanding")
        
        st.subheader("Résultat du Calcul DCF")
        st.write(f"**Valeur Intrinsèque (Entreprise) : {intrinsic_value:,.2f}**")
        if intrinsic_value_per_share is not None:
            st.write(f"**Valeur Intrinsèque par Action : {intrinsic_value_per_share:,.2f}**")
    else:
        st.error("Le Free Cash Flow doit être supérieur à 0 pour effectuer le calcul de la valeur intrinsèque.")
st.markdown("---")

# ================================
# Prévisions Automatiques et Recommandations
# ================================
st.header("5. Prévisions Automatiques et Recommandations")
with st.spinner("Prévision du cours de l'action..."):
    try:
        hist_data = stock.history(period="2y")
        if not hist_data.empty:
            hist_data = hist_data.reset_index()
            hist_data['Date_ordinal'] = hist_data['Date'].apply(lambda x: x.toordinal())
            X = np.array(hist_data['Date_ordinal']).reshape(-1, 1)
            y_price = hist_data['Close'].values
            model = LinearRegression()
            model.fit(X, y_price)
            last_date = hist_data['Date'].max()
            future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
            future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
            predictions = model.predict(future_ordinals)

            # Graphique de prévision du cours de l'action amélioré
            fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
            ax_pred.plot(hist_data['Date'], y_price, marker='o', markersize=4, label="Historique")
            ax_pred.plot(future_dates, predictions, linestyle="--", linewidth=2, label="Prévisions")
            ax_pred.set_xlabel("Date", fontsize=12)
            ax_pred.set_ylabel("Prix de clôture", fontsize=12)
            ax_pred.set_title("Prévision du cours de l'action sur 30 jours", fontsize=14, fontweight="bold")
            ax_pred.legend(fontsize=10)
            ax_pred.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig_pred)

            current_price = y_price[-1]
            predicted_price = predictions[-1]
            pct_change = ((predicted_price - current_price) / current_price) * 100
            st.write(f"Variation prévue sur 30 jours : **{pct_change:.2f}%**")

            # Recommandation présentée via une boîte attractive
            if pct_change > 5:
                rec_forecast = "Tendance haussière détectée. Recommandation : **Acheter**."
            elif pct_change < -5:
                rec_forecast = "Tendance baissière détectée. Recommandation : **Vendre** ou **Attendre**."
            else:
                rec_forecast = "Tendance stable. Recommandation : **Conserver** ou surveiller."
            st.markdown("#### Recommandation basée sur la prévision du cours")
            st.success(rec_forecast)
        else:
            st.warning("Pas de données historiques disponibles pour la prévision.")
    except Exception as e:
        st.error("Erreur lors de la prévision : " + str(e))
st.markdown("---")

# ================================
# Analyse des Forces et Faiblesses de l'Entreprise
# ================================
st.header("6. Analyse des Forces et Faiblesses de l'Entreprise")
company_ratios = {
    "P/E": user_pe,
    "Dividend Yield": user_dividend_yield,
    "EV/EBITDA": user_ev_ebitda,
    "Free Cash Flow Yield": user_fcf_yield,
    "Marge Nette": user_net_margin,
    "Debt Ratio": data.get("debt_ratio")
}

analysis_data = []

# Analyse pour chaque ratio avec interprétation
# P/E
pe_value = user_pe
if pe_value > industry_pe:
    interpretation_pe = "La valorisation est supérieure à la norme industrielle, indiquant une possible surévaluation."
elif min_pe_threshold <= pe_value <= industry_pe:
    interpretation_pe = "La valorisation est attractive par rapport à l'industrie."
elif pe_value < min_pe_threshold:
    interpretation_pe = "La valorisation est très basse par rapport au seuil minimum, ce qui pourrait signaler des risques."
else:
    interpretation_pe = ""
analysis_data.append({
    "Ratio": "P/E",
    "Valeur Entreprise": pe_value,
    "Norme Industrie": industry_pe,
    "Seuil Minimum Acceptable": min_pe_threshold,
    "Interprétation": interpretation_pe
})

# Dividend Yield
div_yield_value = user_dividend_yield
if div_yield_value < min_dividend_yield_threshold:
    interpretation_div = "Le rendement des dividendes est inférieur au seuil minimal, ce qui représente une faiblesse."
elif min_dividend_yield_threshold <= div_yield_value < industry_dividend_yield:
    interpretation_div = "Le rendement des dividendes est inférieur à la norme industrielle."
elif div_yield_value >= industry_dividend_yield:
    interpretation_div = "Le rendement des dividendes est supérieur ou égal à la norme industrielle, ce qui est positif."
else:
    interpretation_div = ""
analysis_data.append({
    "Ratio": "Dividend Yield",
    "Valeur Entreprise": div_yield_value,
    "Norme Industrie": industry_dividend_yield,
    "Seuil Minimum Acceptable": min_dividend_yield_threshold,
    "Interprétation": interpretation_div
})

# EV/EBITDA
ev_ebitda_value = user_ev_ebitda
if ev_ebitda_value > industry_ev_ebitda:
    interpretation_ev = "Le multiple EV/EBITDA est supérieur à la norme industrielle, ce qui peut indiquer une surévaluation."
elif min_ev_ebitda_threshold <= ev_ebitda_value <= industry_ev_ebitda:
    interpretation_ev = "Le multiple EV/EBITDA est attractif par rapport à l'industrie."
elif ev_ebitda_value < min_ev_ebitda_threshold:
    interpretation_ev = "Le multiple EV/EBITDA est très faible par rapport au seuil minimal, ce qui pourrait être inquiétant."
else:
    interpretation_ev = ""
analysis_data.append({
    "Ratio": "EV/EBITDA",
    "Valeur Entreprise": ev_ebitda_value,
    "Norme Industrie": industry_ev_ebitda,
    "Seuil Minimum Acceptable": min_ev_ebitda_threshold,
    "Interprétation": interpretation_ev
})

# Free Cash Flow Yield
fcf_yield_value = user_fcf_yield
if fcf_yield_value < min_fcf_yield_threshold:
    interpretation_fcf = "Le rendement du Free Cash Flow est inférieur au seuil minimal, représentant une faiblesse."
elif min_fcf_yield_threshold <= fcf_yield_value < industry_fcf_yield:
    interpretation_fcf = "Le rendement du Free Cash Flow est inférieur à la norme industrielle."
elif fcf_yield_value >= industry_fcf_yield:
    interpretation_fcf = "Le rendement du Free Cash Flow est supérieur ou égal à la norme industrielle, ce qui est favorable."
else:
    interpretation_fcf = ""
analysis_data.append({
    "Ratio": "Free Cash Flow Yield",
    "Valeur Entreprise": fcf_yield_value,
    "Norme Industrie": industry_fcf_yield,
    "Seuil Minimum Acceptable": min_fcf_yield_threshold,
    "Interprétation": interpretation_fcf
})

# Marge Nette
net_margin_value = user_net_margin
if net_margin_value < min_net_margin_threshold:
    interpretation_net = "La marge nette est inférieure au seuil minimal, indiquant une faiblesse."
elif min_net_margin_threshold <= net_margin_value < industry_net_margin:
    interpretation_net = "La marge nette est en dessous de la norme industrielle."
elif net_margin_value >= industry_net_margin:
    interpretation_net = "La marge nette est au moins égale ou supérieure à la norme industrielle, signe de performance."
else:
    interpretation_net = ""
analysis_data.append({
    "Ratio": "Marge Nette",
    "Valeur Entreprise": net_margin_value,
    "Norme Industrie": industry_net_margin,
    "Seuil Minimum Acceptable": min_net_margin_threshold,
    "Interprétation": interpretation_net
})

# Debt Ratio
debt_ratio_value = data.get("debt_ratio")
if debt_ratio_value is None:
    interpretation_debt = "Donnée non disponible."
else:
    if debt_ratio_value > industry_debt_ratio:
        interpretation_debt = "Le ratio d'endettement est supérieur à la norme industrielle, ce qui peut être risqué."
    elif min_debt_ratio_threshold <= debt_ratio_value <= industry_debt_ratio:
        interpretation_debt = "Le ratio d'endettement est acceptable par rapport à l'industrie."
    elif debt_ratio_value < min_debt_ratio_threshold:
        interpretation_debt = "Le ratio d'endettement est très faible, ce qui est généralement positif."
    else:
        interpretation_debt = ""
analysis_data.append({
    "Ratio": "Debt Ratio",
    "Valeur Entreprise": debt_ratio_value,
    "Norme Industrie": industry_debt_ratio,
    "Seuil Minimum Acceptable": min_debt_ratio_threshold,
    "Interprétation": interpretation_debt
})

df_analysis = pd.DataFrame(analysis_data).drop(columns=["Interprétation"])
st.dataframe(df_analysis)
st.markdown("#### Interprétations détaillées :")
for item in analysis_data:
    st.markdown(f"**{item['Ratio']}** : {item['Interprétation']}")
