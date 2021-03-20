#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>

template <typename T>
std::vector<T> read_binary_file(const char *fname, size_t num = 0) {
  std::vector<T> vec;
  std::ifstream ifs(fname, std::ios::in | std::ios::binary);
  if (ifs.good()) {
    ifs.unsetf(std::ios::skipws);
    std::streampos file_size;
    ifs.seekg(0, std::ios::end);
    file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    size_t max_num = file_size / sizeof(T);
    vec.resize(num ? (std::min)(max_num, num) : max_num);
    ifs.read(reinterpret_cast<char *>(vec.data()), vec.size() * sizeof(T));
  }
  return vec;
}

std::vector<char> read_pgm_image(const char *fname) {
  std::vector<char> result;
  std::ifstream ifs(fname);
  if (ifs.good()) {
    int numrows, numcols, max_pixel;
    std::string line;

    std::getline(ifs, line);
    if (line != "P5") {
      std::cerr << "Not a PGM image\n";
      return result;
    }

    while (std::getline(ifs, line)) {
      if (line[0] != '#') {
        break;
      }
    }

    std::stringstream ss(line);

    ss >> numcols >> numrows;
    result.resize(numrows * numcols);

    while (std::getline(ifs, line)) {
      if (line[0] != '#') {
        break;
      }
    }

    ss = std::stringstream(line);
    ss >> max_pixel;

    if (max_pixel != 255) {
      return result;
    }

    std::cout << "Rows " << numrows << ", cols " << numcols << ", max_pixel = " << max_pixel << std::endl;

    ifs.read(&result[0], numrows * numcols);
  }
  return result;
}

bool cmp_binary_files(const char *fname1, const char *fname2, float tolerance) {
  const auto vec1 = read_pgm_image(fname1);
  const auto vec2 = read_pgm_image(fname2);
  if (vec1.size() != vec2.size()) {
    std::cerr << fname1 << " size is " << vec1.size();
    std::cerr << " whereas " << fname2 << " size is " << vec2.size()
              << std::endl;
    return false;
  }
  size_t allowed_pixels = (tolerance) / 100. * vec1.size();
  size_t current_pixels = 0;
  for (size_t i = 0; i < vec1.size(); i++) {
    if (vec1[i] != vec2[i]) {
      current_pixels++;
    } 
  }

  if (current_pixels > allowed_pixels) {
    std::cerr << "There are two many differences" << std::endl;
    std::cerr << "Allowed different pixels: " << allowed_pixels << ", different pixels in this file " << current_pixels << std::endl;
    return false;
  }
  return true;
}

int main(int argc, char* argv[]) {
    
    bool result = cmp_binary_files(argv[1], argv[2], 0.01);

    if (!result) {
      std::cout << "Validation failed" << std::endl;        
      return 1;
    }
        
    std::cout << "Validation successful" << std::endl;
    return 0;
}
