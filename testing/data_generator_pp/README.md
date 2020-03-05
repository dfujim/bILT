# ```data_generator_pp```: simulate time-differential β-NMR using ```C++```!

To build the simulator simply run:
```bash
make
```
Note that building requires the following libraries:
- [```ROOT```](https://github.com/root-project/root)
- [```yaml-cpp```](https://github.com/jbeder/yaml-cpp)

A simulation can be run using:
```bash
./data_generator_pp config.yaml
```
where the important details (e.g, number of probes, histogram binning, etc.) are defined in ```config.yaml```.

The results are saved as pre-binned histograms to a ```.root``` file (both specified in ```config.yaml```). These can easily be inspected using ROOT's interpreter:
```c++
root -l
root [0] auto file = TFile::Open("result.root")
root [1] .ls // list all the saved objects
root [2] hsN->Draw("nostack plc") // draw the four raw histograms
root [3] hsA->Draw("nostack") // draw the asymmtry in both helicities
root [4] hA->Draw() // draw the 4-counter combined asymmetry
```

The output root file can be translated to a csv using the following command: 

```bash
python3 output2csv.py filename.root
```

The resulting file name has the format `filename.csv`.
