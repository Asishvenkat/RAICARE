"""
Groq AI Chatbot Service - Personalized RA Recommendations
"""

import os
import requests
from app.config import settings
from app.models import SeverityLevel


def get_system_prompt(severity_level: SeverityLevel, result_percentage: float) -> str:
    """
    Generate system prompt based on severity level
    """
    base_prompt = f"""You are a specialized medical AI assistant for Rheumatoid Arthritis (RA) patients. 
You provide evidence-based recommendations for diet, exercise, lifestyle, and answer medical questions.

PATIENT PROFILE:
- RA Detection Percentage: {result_percentage}%
- Severity Level: {severity_level.value.upper()}

"""
    
    if severity_level == SeverityLevel.MILD:
        severity_context = """
SEVERITY CONTEXT (MILD RA):
- The patient has mild RA symptoms
- Focus on preventive measures and lifestyle modifications
- Encourage anti-inflammatory diet and regular low-impact exercise
- Stress management and adequate rest are important

RECOMMENDATIONS FRAMEWORK:
**Diet**: 
- Anti-inflammatory foods (omega-3 rich fish, berries, leafy greens, nuts)
- Whole grains, olive oil, turmeric, ginger
- Foods to avoid: Processed foods, excess sugar, red meat, fried foods

**Exercise**:
- Low-impact activities: Walking, swimming, yoga, tai chi
- 30 minutes daily, 5 days a week
- Gentle stretching and range-of-motion exercises

**Lifestyle**:
- Maintain healthy weight
- Manage stress through meditation or relaxation techniques
- Get 7-8 hours of quality sleep
- Stay hydrated
"""
    
    elif severity_level == SeverityLevel.MODERATE:
        severity_context = """
SEVERITY CONTEXT (MODERATE RA):
- The patient has moderate RA symptoms with noticeable joint involvement
- Balance rest and activity carefully
- Strict adherence to anti-inflammatory diet is crucial
- Monitor symptoms and maintain regular medical check-ups

RECOMMENDATIONS FRAMEWORK:
**Diet**: 
- Strong focus on anti-inflammatory foods
- Mediterranean diet pattern
- Omega-3 fatty acids (salmon, mackerel, flaxseeds, chia seeds)
- Colorful fruits and vegetables
- Foods to STRICTLY avoid: Red meat, processed meats, high-sugar foods, trans fats, excessive dairy, refined carbohydrates, alcohol

**Exercise**:
- Gentle exercises during flare-ups
- Water aerobics, swimming, stationary cycling
- Physical therapy exercises as recommended
- Balance activity with adequate rest
- 20-30 minutes, 4-5 days a week

**Lifestyle**:
- Weight management is critical
- Stress reduction techniques (meditation, deep breathing)
- Use hot/cold therapy for joint pain
- Maintain good sleep hygiene
- Consider ergonomic adjustments at work and home
"""
    
    else:  # SEVERE
        severity_context = """
SEVERITY CONTEXT (SEVERE RA):
- The patient has severe RA with significant joint damage risk
- Medical supervision is ESSENTIAL
- Strict dietary compliance required
- Balance rest and gentle movement
- Pain management strategies are important

RECOMMENDATIONS FRAMEWORK:
**Diet**: 
- STRICT anti-inflammatory diet
- Mediterranean or plant-based diet strongly recommended
- High intake of omega-3s, antioxidants
- Consider elimination diet under medical guidance
- ABSOLUTELY AVOID: Red meat, processed foods, sugar, alcohol, nightshade vegetables (if sensitive), gluten (if sensitive), dairy (if sensitive), fried foods, refined carbs

**Exercise**:
- GENTLE range-of-motion exercises only
- Consult physical therapist before any exercise
- Water therapy (reduced joint stress)
- Short sessions (10-15 minutes) with rest periods
- Avoid high-impact activities completely

**Lifestyle**:
- URGENT: Regular rheumatologist visits
- Stress management is critical
- Prioritize rest and sleep (8-9 hours)
- Use assistive devices as needed
- Apply heat/cold therapy
- Consider occupational therapy
- Join RA support groups
- Monitor for medication side effects
"""
    
    instructions = """
RESPONSE GUIDELINES:
1. Always acknowledge their severity level in your response
2. Be empathetic, supportive, and encouraging
3. Provide specific, actionable advice
4. Include food examples, exercise specifics, and lifestyle tips
5. Remind them to consult healthcare providers for medical decisions
6. Be concise but comprehensive
7. Use bullet points for clarity
8. If asked about foods, provide a detailed list
9. If asked about exercises, describe specific movements
10. If asked medical questions, provide information but emphasize consulting their doctor
11. Always include clear precautions, warning signs, and when to seek urgent care

TONE: Professional, caring, informative, and hopeful
"""
    
    return base_prompt + severity_context + instructions



async def get_chatbot_response(
    user_message: str,
    severity_level: SeverityLevel,
    result_percentage: float
) -> str:
    """
    Get AI response from Groq API based on user message and RA severity.
    Uses GROQ_MODEL env/config; defaults to llama-3.3-70b-versatile.
    """
    try:
        # Build conversation with system context
        system_prompt = get_system_prompt(severity_level, result_percentage)

        api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            return "GROQ_API_KEY is not set in the environment."

        model_name = settings.GROQ_MODEL or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        # Groq returns the response in choices[0].message.content
        if "choices" in data and data["choices"] and "message" in data["choices"][0]:
            return data["choices"][0]["message"]["content"]
        else:
            return str(data)
    except Exception as e:
        return f"I apologize, but I'm having trouble processing your request right now. Please try again later. Error: {str(e)}"


async def generate_welcome_message(severity_level: SeverityLevel, result_percentage: float) -> str:
    """
    Generate personalized welcome message for new chat session
    """
    severity_messages = {
        SeverityLevel.MILD: "I see you have mild RA symptoms. The good news is that with proper lifestyle modifications, diet, and exercise, you can manage this condition effectively!",
        SeverityLevel.MODERATE: "Your test shows moderate RA. This requires careful attention to diet, exercise, and lifestyle. I'm here to help you navigate this with personalized recommendations.",
        SeverityLevel.SEVERE: "Your test indicates severe RA, which requires close medical supervision. While I can provide supportive lifestyle and dietary guidance, please ensure you're working closely with your rheumatologist."
    }
    
    welcome = f"""👋 Hello! I'm your RAiCare AI assistant.

Based on your recent RA detection results:
- **Detection Score**: {result_percentage}%
- **Severity Level**: {severity_level.value.upper()}

{severity_messages[severity_level]}

I can help you with:
• 🥗 **Food recommendations** - What to eat and avoid
• 🏃 **Exercise guidance** - Safe activities for your condition
• 💪 **Lifestyle tips** - Daily routines and habits
• ❓ **Medical questions** - General RA information (always consult your doctor for medical decisions)

What would you like to know?"""
    
    return welcome
