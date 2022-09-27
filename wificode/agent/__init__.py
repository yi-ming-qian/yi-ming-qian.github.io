from agent.agent_wifi import WifiAgent

def get_agent(config):
    if config.module == 'wifi':
        return WifiAgent(config)
    else:
        raise ValueError

