import sys
import triton_llm_demo
from okahu_apptrace.instrumentor import setup_okahu_telemetry
from credential_utilties.environment import setOkahuEnvironmentVariablesFromConfig
import logging


def main():
    #Okahu instrumentation
    setOkahuEnvironmentVariablesFromConfig(sys.argv[1])
    setup_okahu_telemetry(workflow_name="llama_triton_wf4")

    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    #invoke the underlying application
    triton_llm_demo.main()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage " + sys.argv[0] + " <config-file-path>")
        sys.exit(1)
    main()
