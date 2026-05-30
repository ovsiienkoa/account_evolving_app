import streamlit as st
import json
import uuid
import sys
import os
import hashlib
import hmac
from dotenv import dotenv_values

from receipt_processing_agent.agent import ReceiptProcessingAgent
from data_analytics_agent.agent import DataAnalyticsAgent

config = dotenv_values(".env")
receipt_agent = ReceiptProcessingAgent(config)
data_analytics_agent = DataAnalyticsAgent(config)
def main():
    st.set_page_config(page_title="Account Evolving App", layout="wide")

    if "user_id" not in st.session_state:
        st.session_state.user_id = None

    if not getattr(st, "user", None) or not st.user.is_logged_in:
        st.title("Welcome to Account Evolving App")
        st.write("Please log in with your Google account to continue.")
        if st.button("Login with Google"):
            st.login("google")
    else:
        if not st.session_state.user_id:
            google_id = st.user.get("sub")
            if google_id:
                secret = config.get("USER_SECRET", "")
                hashed_id = hmac.new(secret.encode('utf-8'), google_id.encode('utf-8'), hashlib.sha256).hexdigest()
                st.session_state.user_id = hashed_id
            else:
                st.error("Could not retrieve user ID from Google.")

        st.sidebar.title(f"User: {st.session_state.user_id}")
        if st.sidebar.button("Logout"):
            st.session_state.user_id = None
            st.logout()

        st.title("Agent Chats")

        agent_choice = st.radio("Select Agent", ["Receipt Processing Agent", "Data Analytics Agent"], horizontal=True)

        chat_key = f"chat_history_{agent_choice}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        history = st.session_state[chat_key]

        # Display history
        for msg in history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "data" in msg:
                    st.json(msg["data"])
                if "media" in msg and msg["media"]:
                    st.image(msg["media"], caption="Uploaded Image")
                if "plot_config" in msg and msg["plot_config"]:
                    fig = data_analytics_agent.make_plot(msg["plot_config"], msg.get("sql_output"))
                    if fig:
                        st.plotly_chart(fig)

        if agent_choice == "Receipt Processing Agent":
            if st.session_state.get("pending_receipt_data"):
                st.info("You have a pending receipt to review.")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Approve & Commit to Database", type="primary"):
                        success = receipt_agent.commit_receipt(st.session_state.pending_receipt_data)
                        if success:
                            st.success("Receipt committed successfully!")
                            st.session_state[chat_key].append({"role": "assistant", "content": "Receipt committed successfully!"})
                        else:
                            st.error("Failed to commit receipt.")
                        st.session_state.pending_receipt_data = None
                        st.rerun()
                with col2:
                    with st.form("feedback_form"):
                        feedback = st.text_input("Or provide feedback to fix it:")
                        submitted = st.form_submit_button("Reject & Fix")
                        if submitted and feedback:
                            with st.spinner("Applying feedback..."):
                                st.session_state[chat_key].append({"role": "user", "content": f"Feedback: {feedback}"})
                                response = receipt_agent.fix_receipt_data(st.session_state.pending_receipt_data, feedback)
                                st.session_state.pending_receipt_data = response["data"]
                                st.session_state[chat_key].append({"role": "assistant", "content": response["text"], "data": response["data"]})
                                st.rerun()

            uploaded_file = st.file_uploader("Upload Receipt (Image)", type=["png", "jpg", "jpeg"])
        elif agent_choice == "Data Analytics Agent":
            if st.session_state.get("pending_analysis_data"):
                st.info("You have a pending analysis to review.")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Approve & Log", type="primary"):
                        pending = st.session_state.pending_analysis_data
                        data_analytics_agent.commit_to_history(
                            user_query=pending["user_query"],
                            sql_query=pending["sql_query"],
                            analysis=pending["analysis"]
                        )
                        st.success("Analysis logged to history table successfully!")
                        st.session_state[chat_key].append({"role": "assistant", "content": "Analysis logged to history table successfully!"})
                        st.session_state.pending_analysis_data = None
                        st.rerun()
                with col2:
                    with st.form("analysis_feedback_form"):
                        feedback = st.text_input("Or provide feedback to fix it:")
                        submitted = st.form_submit_button("Reject & Fix")
                        if submitted and feedback:
                            with st.spinner("Applying feedback..."):
                                st.session_state[chat_key].append({"role": "user", "content": f"Feedback: {feedback}"})
                                pending = st.session_state.pending_analysis_data
                                analysis = data_analytics_agent.format_answer(
                                    user_query=pending["user_query"],
                                    sql_query=pending["sql_query"],
                                    sql_output=pending["sql_output"],
                                    feedback=feedback
                                )
                                pending["analysis"] = analysis
                                st.session_state.pending_analysis_data = pending
                                
                                agent_msg = {"role": "assistant", "content": analysis.get("text_response", "Updated analysis.")}
                                if analysis.get("plot_config"):
                                    agent_msg["plot_config"] = analysis["plot_config"]
                                    agent_msg["sql_output"] = pending["sql_output"]
                                
                                st.session_state[chat_key].append(agent_msg)
                                st.rerun()
            uploaded_file = None

        prompt = st.chat_input("Type your message here...")

        if prompt or (uploaded_file and st.button("Submit Upload")):
            user_content = prompt if prompt else "Uploaded a file."
            
            with st.chat_message("user"):
                st.write(user_content)
                if uploaded_file:
                    st.image(uploaded_file)
            
            user_msg = {"role": "user", "content": user_content}
            if uploaded_file:
                user_msg["media"] = uploaded_file.getvalue()
                
            st.session_state[chat_key].append(user_msg)

            with st.chat_message("assistant"):
                if agent_choice == "Receipt Processing Agent":
                    if uploaded_file or prompt:
                        response = receipt_agent.process_receipt(
                            image_bytes=uploaded_file.getvalue() if uploaded_file else None, 
                            mime_type=uploaded_file.type if uploaded_file else None, 
                            text_input=prompt if not uploaded_file else None,
                            user_id=st.session_state.user_id
                        )
                        st.session_state.pending_receipt_data = response.get("data")
                        agent_msg = {"role": "assistant", "content": response["text"]}
                        st.write(response["text"])
                    else:
                        agent_msg = {"role": "assistant", "content": "Please provide a receipt image or text."}
                        st.write(agent_msg["content"])
                else:
                    if prompt:
                        try:
                            with st.spinner("Generating and executing query..."):
                                response, sql_output = data_analytics_agent.generate_and_execute_sql(prompt, user_id=st.session_state.user_id)
                            
                            sql_query = response["data"]["generated_sql"]
                            st.write(response["text"])
                                
                            with st.spinner("Analyzing results..."):
                                analysis = data_analytics_agent.format_answer(prompt, sql_query, sql_output)
                                
                            st.session_state.pending_analysis_data = {
                                "user_query": prompt,
                                "sql_query": sql_query,
                                "sql_output": sql_output,
                                "analysis": analysis
                            }
                            
                            agent_msg = {"role": "assistant", "content": analysis.get("text_response", "Analysis complete.")}
                            if analysis.get("plot_config"):
                                agent_msg["plot_config"] = analysis["plot_config"]
                                agent_msg["sql_output"] = sql_output
                                
                            st.write(agent_msg["content"])
                            if "plot_config" in agent_msg and agent_msg["plot_config"]:
                                fig = data_analytics_agent.make_plot(agent_msg["plot_config"], agent_msg.get("sql_output"))
                                if fig:
                                    st.plotly_chart(fig)
                        except ValueError as e:
                            st.error(str(e))
                            agent_msg = {"role": "assistant", "content": str(e)}
                    else:
                        agent_msg = {"role": "assistant", "content": "Please ask a question about your spending data."}
                        st.write(agent_msg["content"])
                        
            st.session_state[chat_key].append(agent_msg)
            
            # Rerun so the UI updates immediately and shows the Approve/Reject buttons
            st.rerun()

if __name__ == "__main__":
    main()
