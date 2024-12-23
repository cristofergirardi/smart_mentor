class PromptSelfVerification():
    
    @property
    def self_verification(self):
        return f'''
            Please rewrite the question and answer to give better response.
        '''