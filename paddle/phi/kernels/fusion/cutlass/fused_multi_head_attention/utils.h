#include "cutlass/platform/platform.h"
namespace cutlass {
namespace platform {

// template< class To, class From >
// constexpr To CUTLASS_HOST_DEVICE bit_cast(const From& from ) noexcept;

// template <class To, class From>
// constexpr To CUTLASS_HOST_DEVICE bit_cast(const From& src) noexcept
// {
//   static_assert(sizeof(To) == sizeof(From), "sizes must match");
//   return reinterpret_cast<To const &>(src);
// }

// template <>
// struct numeric_limits<float> {
//   CUTLASS_HOST_DEVICE
//   static constexpr float infinity() noexcept { return bit_cast<float, int32_t>(0x7f800000);}
//   static constexpr bool is_integer = false;
//   static constexpr bool has_infinity = true;
// };

// template <>
// struct numeric_limits<cutlass::half_t> {
//   CUTLASS_HOST_DEVICE
//   static const cutlass::half_t infinity() noexcept { return bit_cast<cutlass::half_t, int16_t>(0x7800);}
//   static constexpr bool is_integer = false;
//   static constexpr bool has_infinity = true;
// };

}  // namespace platform
}  // namespace cutlass
