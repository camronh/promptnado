def format_rules(rules):
    rule_strs = [
        f"<ATTEMPT>\nReasoning: {rule['reasoning']}\nPrompt: '{rule['prompt']}'\n</ATTEMPT>" for rule in rules]

    attempts_str = '\n'.join(rule_strs)

    return f"""Here are some prompts we have already tried:
    
<ATTEMPTS>
{attempts_str}
</ATTEMPTS>
"""
