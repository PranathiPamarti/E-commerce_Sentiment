import streamlit as st

st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

st.title("E-Commerce Reviews & Recommendation Demo")

st.markdown("""
Explore a generated review dataset, compare product feedback, and try an
experimental hybrid recommender.

The included data is synthetic—not real customer feedback. Recommendation
metrics are provided as a reproducible pipeline sanity check, not proof of
real-world performance.
""")

st.info("Choose a page from the sidebar to explore insights, products, and recommendations.")
