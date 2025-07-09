import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import io
import os
import sys

# Add error handling for imports
try:
    from agent import agent
except ImportError as e:
    st.error(f"Error importing agent: {e}")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Inventory Prediction Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
    .stButton > button {
        font-size: 1.1rem;
        font-weight: bold;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stAlert {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Health check for API keys
def check_api_keys():
    """Check if all required API keys are present"""
    required_keys = ['TAVILY_API_KEY', 'OPENAI_API_KEY', 'GOOGLE_API_KEY']
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        st.error(f"Missing API keys: {missing_keys}")
        st.info("Please set these environment variables on your EC2 instance")
        return False
    return True

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'agent_ready' not in st.session_state:
    st.session_state.agent_ready = False

# Main title
st.markdown('<h1 class="main-header">📦 Inventory Prediction Agent</h1>', unsafe_allow_html=True)

# Check system status
with st.expander("🔍 System Status", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        api_status = check_api_keys()
        if api_status:
            st.success("✅ API Keys: OK")
        else:
            st.error("❌ API Keys: Missing")
    
    with col2:
        try:
            if agent:
                st.success("✅ Agent: Ready")
                st.session_state.agent_ready = True
            else:
                st.error("❌ Agent: Not initialized")
        except Exception as e:
            st.error(f"❌ Agent: Error - {str(e)}")
    
    with col3:
        # Check model files
        model_files = [
            'xgboost_inventory_model.pkl',
            'label_encoders.pkl',
            'feature_columns.pkl',
            'synthetic_retail_sales_data.csv'
        ]
        missing_files = [f for f in model_files if not os.path.exists(f)]
        if not missing_files:
            st.success("✅ Model Files: OK")
        else:
            st.error(f"❌ Model Files: Missing {missing_files}")

# Only show main interface if agent is ready
if not st.session_state.agent_ready:
    st.warning("⚠️ Please fix the issues above before proceeding")
    st.stop()

# Sidebar for configuration
with st.sidebar:
    st.header("🛠️ Configuration")
    
    # Server info
    st.info(f"**Server Info:**\n"
            f"- Python: {sys.version.split()[0]}\n"
            f"- Working Dir: {os.getcwd()}\n"
            f"- Host: {os.getenv('HOSTNAME', 'EC2')}")
    
    # Quick action buttons
    st.subheader("Quick Actions")
    
    if st.button("🌤️ Get Weather Forecast"):
        st.session_state.query_template = "Get weather forecast for {city} on {date}"
        
    if st.button("🎉 Check Holidays"):
        st.session_state.query_template = "Check holidays on {date} in India"
        
    if st.button("📊 Predict Inventory"):
        st.session_state.query_template = "Predict inventory for {product_id} on {date} in {city} considering all events"
    
    # Example queries
    st.subheader("📝 Example Queries")
    examples = [
        "Find inventory for SKU010 on 2025-07-15 in Chennai",
        "Get weather forecast for Chennai from 2025-07-10 to 2025-07-15",
        "Check holidays on 2025-08-15 in India",
        "Search economic events in Mumbai",
        "Predict inventory on 12th July 2025 in Chennai for SKU010 considering weather, disasters, and holidays"
    ]
    
    for i, example in enumerate(examples):
        if st.button(f"Example {i+1}", key=f"example_{i}"):
            st.session_state.user_query = example

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Ask Your Question")
    
    # Input methods
    input_method = st.radio("Choose input method:", ["Free Text", "Structured Form"])
    
    if input_method == "Free Text":
        user_query = st.text_area(
            "Enter your query:",
            value=st.session_state.get('user_query', ''),
            height=100,
            placeholder="e.g., Find inventory for SKU010 on 2025-07-15 in Chennai considering all events"
        )
    else:
        # Structured form
        st.write("Fill out the form below:")
        
        col_form1, col_form2 = st.columns(2)
        
        with col_form1:
            action = st.selectbox("Action:", [
                "Predict Inventory",
                "Get Weather Forecast", 
                "Check Holidays",
                "Search Economic Events",
                "Search Disaster Events",
                "Search Weather Events"
            ])
            
            city = st.text_input("City:", value="Chennai")
            date = st.date_input("Date:", value=datetime.now() + timedelta(days=1))
            
        with col_form2:
            if action == "Predict Inventory":
                product_id = st.text_input("Product ID:", value="SKU010")
                consider_events = st.checkbox("Consider all events (weather, disasters, holidays)", value=True)
            
            if action == "Get Weather Forecast":
                end_date = st.date_input("End Date (optional):", value=date)
        
        # Generate query from form
        if action == "Predict Inventory":
            events_text = "considering all events (weather, disasters, holidays, economic events)" if consider_events else ""
            user_query = f"Predict inventory for {product_id} on {date} in {city} {events_text}"
        elif action == "Get Weather Forecast":
            if end_date != date:
                user_query = f"Get weather forecast for {city} from {date} to {end_date}"
            else:
                user_query = f"Get weather forecast for {city} on {date}"
        elif action == "Check Holidays":
            user_query = f"Check holidays on {date} in India"
        elif action == "Search Economic Events":
            user_query = f"Search economic events in {city}"
        elif action == "Search Disaster Events":
            user_query = f"Search disaster events in {city}"
        elif action == "Search Weather Events":
            user_query = f"Search weather events in {city}"
    
    # Process query
    if st.button("🚀 Process Query", type="primary"):
        if user_query.strip():
            with st.spinner("Processing your request..."):
                try:
                    # Add timeout handling for EC2
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Request timed out")
                    
                    # Set timeout to 60 seconds
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(60)
                    
                    try:
                        response = agent.process_query(user_query)
                        signal.alarm(0)  # Cancel the alarm
                        
                        st.session_state.last_response = response
                        st.session_state.last_query = user_query
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'query': user_query,
                            'response': response,
                            'timestamp': datetime.now()
                        })
                        
                    except TimeoutError:
                        st.error("Request timed out. Please try again with a simpler query.")
                        st.session_state.last_response = None
                        
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.session_state.last_response = None
                    
                    # Log error for debugging
                    st.error(f"Debug info: {type(e).__name__}: {str(e)}")
        else:
            st.warning("Please enter a query!")

with col2:
    st.subheader("📈 Quick Stats")
    
    # Display some stats or information
    st.info("**Agent Capabilities:**\n"
            "- Weather forecasting\n"
            "- Holiday detection\n"
            "- Economic event tracking\n"
            "- Disaster monitoring\n"
            "- Inventory prediction")
    
    # Resource usage (if available)
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        st.metric("CPU Usage", f"{cpu_percent}%")
        st.metric("Memory Usage", f"{memory_percent}%")
    except ImportError:
        st.info("Install psutil for resource monitoring")
    
    # Show graph visualization
    if st.button("🔍 View Agent Graph"):
        try:
            graph_png = agent.get_graph_visualization()
            if graph_png:
                st.image(graph_png, caption="Agent Workflow Graph")
            else:
                st.warning("Graph visualization not available")
        except Exception as e:
            st.error(f"Error generating graph: {e}")

# Display response
if hasattr(st.session_state, 'last_response') and st.session_state.last_response:
    st.subheader("🤖 Agent Response")
    
    # Show the query that was processed
    st.write(f"**Query:** {st.session_state.last_query}")
    
    # Show the response in a styled box
    st.markdown(f'<div class="response-box">{st.session_state.last_response}</div>', unsafe_allow_html=True)
    
    # Option to download response
    if st.button("💾 Download Response"):
        response_data = f"Query: {st.session_state.last_query}\n\nResponse: {st.session_state.last_response}"
        st.download_button(
            label="Download as Text",
            data=response_data,
            file_name=f"inventory_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# Chat history
st.subheader("💬 Chat History")

# Display chat history
if st.session_state.chat_history:
    for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5 chats
        with st.expander(f"Chat {len(st.session_state.chat_history) - i}: {chat['query'][:50]}..."):
            st.write(f"**Time:** {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Query:** {chat['query']}")
            st.write(f"**Response:** {chat['response']}")
else:
    st.info("No chat history yet. Start by asking a question!")

# Clear chat history
if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit, LangChain, and LangGraph | Running on EC2*")
