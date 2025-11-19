### Steps to run the plotting script:

1. Checkout to the `dev/log_parser` branch.
2. Update log_parser.py with log file:
    - In the `configure()` function, start making the updates:
        - Populate `log_file` with the aggregator log file path
        - Populate `suffix` to describe the run variation. e.g - reject_stale, keep_stale, ...
        - Update `EXPORT_CONFIG...['evaluation_metrics']['default_output_filename']` with a prefix that describes the setup. e.g - async_k10_c50_n150
3. Update comparative_plotter.py with parsed file:
    1. Make the following updates at the top of the file:
        - Update value in `SYSTEMS_DATA`
            - Key: The text that you want to display in the plot. 
            - Value: The run logs of the system. Multiple logs per system/ setup because there can be a difference in the accuracy values between 2 runs of FwdLLMFlame.
4. Run commands:
 - ```python log_parser.py```
 - ```python comparative_plotter.py```
5. You can see 3 output plots generated in the `plots` directory